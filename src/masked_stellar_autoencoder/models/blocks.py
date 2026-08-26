# loading the packages
import torch
import torch.nn as nn
from rtdl_num_embeddings import (
    PeriodicEmbeddings,
)


def _get_activation(active: str) -> nn.Module:
    if active == "elu":
        return nn.ELU(inplace=True)
    elif active == "gelu":
        return nn.GELU()
    elif active == "relu":
        return nn.ReLU(inplace=True)
    else:
        raise ValueError(
            f"Unsupported activation type: {active}. Use 'elu', 'gelu', or 'relu'"
        )


def _get_norm(norm: str, num_features: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm1d(num_features)
    elif norm == "layer":
        return nn.LayerNorm(num_features)
    else:
        raise ValueError(f"Unsupported norm type: {norm}. Use 'batch' or 'layer'")


class ResBlock(nn.Module):
    """
    Defining an individual residual block as required for ResNets in pytorch
    """

    def __init__(
        self, in_features, out_features, dropout_prob=0.1, active="elu", norm="batch"
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_features, out_features, bias=False)
        self.normal = _get_norm(norm, out_features)
        self.active = _get_activation(active)
        self.dp = nn.Dropout(p=dropout_prob)
        self.lin2 = nn.Linear(out_features, out_features, bias=False)

        if in_features != out_features:
            self.resize = nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                _get_norm(norm, out_features),
            )
        else:
            self.resize = None

    def forward(self, x):
        identity = x

        out = self.lin1(x)
        out = self.normal(out)
        out = self.active(out)
        out = self.dp(out)

        out = self.lin2(out)
        out = self.normal(out)

        if self.resize is not None:
            identity = self.resize(identity)

        out += identity
        out = self.active(out)
        return out


class DenseResnet(nn.Module):
    """
    Fabricates the ResNet, for which the size of the blocks changes according to what is passed to the encoder (decoder is symmetric)
    """

    def __init__(
        self,
        input_dim,
        blocks_dims,
        num_blocks_per_layer=1,
        pe=False,
        d_embedding=8,
        active="elu",
        norm="batch",
    ):
        super().__init__()

        layers = []
        for i, dim in enumerate(blocks_dims):
            if i == 0:
                if pe:
                    layers.append(
                        PeriodicEmbeddings(
                            input_dim, d_embedding=d_embedding, lite=False
                        )
                    )
                    layers.append(nn.Flatten())
                    layers.append(nn.Linear(input_dim * d_embedding, dim))
                    layers.append(ResBlock(dim, dim, active=active, norm=norm))
                else:
                    layers.append(ResBlock(input_dim, dim, active=active, norm=norm))
            else:
                for j in range(num_blocks_per_layer):
                    if j == 0:
                        layers.append(
                            ResBlock(blocks_dims[i - 1], dim, active=active, norm=norm)
                        )
                    else:
                        layers.append(ResBlock(dim, dim, active=active, norm=norm))

        self.dense_resnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_resnet(x)


class TabResnetEncoder(nn.Module):
    """
    Redundant (just calls the class DenseResnet), but matches the shape of the pytorch-widedeep networks for potential model tuning later.
    """

    def __init__(
        self,
        continuous_cols,
        blocks_dims,
        pe_bool=True,
        d_embedding=8,
        active="elu",
        norm="batch",
        cosine_latent: bool = False,
    ):
        super().__init__()
        self.cosine_latent = cosine_latent

        input_dim = continuous_cols  # Length of the data, e.g., 153
        self.encoder = DenseResnet(
            input_dim=input_dim,
            blocks_dims=blocks_dims,
            pe=pe_bool,
            d_embedding=d_embedding,
            active=active,
            norm=norm,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        if self.cosine_latent:
            encoded = nn.functional.normalize(encoded, p=2, dim=-1)
        return encoded


class TabResnet(nn.Module):
    def __init__(
        self,
        continuous_cols,
        blocks_dims,
        output_cols=None,
        d_embedding=8,
        active="elu",
        norm="batch",
        decoder_dims=None,
        cosine_latent: bool = False,
        heteroscedastic: bool = False,
    ):
        super().__init__()

        self.heteroscedastic = heteroscedastic

        self.encoder = TabResnetEncoder(
            continuous_cols=continuous_cols,
            blocks_dims=blocks_dims,
            d_embedding=d_embedding,
            active=active,
            pe_bool=True,
            norm=norm,
            cosine_latent=cosine_latent,
        )

        if decoder_dims is None:
            decoder_dims = blocks_dims[::-1]

        self.decoder = DenseResnet(
            input_dim=blocks_dims[-1],
            blocks_dims=decoder_dims,
            d_embedding=d_embedding,
            active=active,
            pe=False,
            norm=norm,
        )

        if output_cols is None:
            output_cols = continuous_cols
        rec_out = 2 * output_cols if heteroscedastic else output_cols
        self.reconstruction_layer = nn.Linear(decoder_dims[-1], rec_out, bias=False)
        self._output_cols = output_cols

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        out = self.reconstruction_layer(decoded)
        if self.heteroscedastic:
            mean, logvar = out.chunk(2, dim=-1)
            self._last_logvar = logvar
            return mean, encoded
        return out, encoded


# ---------------------------------------------------------------------------
# DenseNet (concatenation skip connections — better for weak-feature regimes)
# ---------------------------------------------------------------------------


class DenseLayer(nn.Module):
    """Single DenseNet layer: BN → ReLU → Linear → Dropout → concat."""

    def __init__(
        self,
        in_features: int,
        growth_rate: int,
        dropout_prob: float = 0.1,
        active: str = "elu",
        norm: str = "batch",
    ):
        super().__init__()
        self.norm = _get_norm(norm, in_features)
        self.act = _get_activation(active)
        self.lin = nn.Linear(in_features, growth_rate)
        self.dp = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.dp(self.lin(self.act(self.norm(x))))], dim=-1)


class DenseBlock(nn.Module):
    """Stack of DenseLayers: each layer concatenates its output with all prior features."""

    def __init__(
        self,
        in_features: int,
        growth_rate: int,
        num_layers: int,
        dropout_prob: float = 0.1,
        active: str = "elu",
        norm: str = "batch",
    ):
        super().__init__()
        layers = []
        feat = in_features
        for _ in range(num_layers):
            layers.append(DenseLayer(feat, growth_rate, dropout_prob, active, norm))
            feat += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_features = feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TabDenseEncoder(nn.Module):
    """PeriodicEmbedding → DenseBlock → linear projection to latent_size."""

    def __init__(
        self,
        input_dim: int,
        latent_size: int,
        growth_rate: int = 64,
        num_layers: int = 8,
        d_embedding: int = 8,
        dropout_prob: float = 0.1,
        active: str = "elu",
        norm: str = "batch",
        cosine_latent: bool = False,
    ):
        super().__init__()
        self.cosine_latent = cosine_latent
        self.pe = PeriodicEmbeddings(input_dim, d_embedding=d_embedding, lite=False)
        self.flatten = nn.Flatten()
        self.input_proj = nn.Linear(input_dim * d_embedding, latent_size)
        self.block = DenseBlock(
            latent_size, growth_rate, num_layers, dropout_prob, active, norm
        )
        self.projection = nn.Linear(self.block.out_features, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(self.pe(x))
        x = self.input_proj(x)
        x = self.block(x)
        x = self.projection(x)
        if self.cosine_latent:
            x = nn.functional.normalize(x, p=2, dim=-1)
        return x


class TabDenseDecoder(nn.Module):
    """DenseBlock decoder (concatenation skip connections)."""

    def __init__(
        self,
        latent_size: int,
        output_dim: int,
        growth_rate: int = 64,
        num_layers: int = 8,
        dropout_prob: float = 0.1,
        active: str = "elu",
        norm: str = "batch",
    ):
        super().__init__()
        self.block = DenseBlock(
            latent_size, growth_rate, num_layers, dropout_prob, active, norm
        )
        self.reconstruction_layer = nn.Linear(
            self.block.out_features, output_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_layer(self.block(x))


class TabDenseNet(nn.Module):
    """Encoder-Decoder with DenseNet blocks (concatenation skip connections).

    Concatenation preserves all feature magnitudes through the network, making
    this architecture better suited than ResNet (addition skips) when individual
    input features carry weak but informative signals.
    """

    def __init__(
        self,
        continuous_cols: int,
        latent_size: int,
        output_cols: int | None = None,
        growth_rate: int = 64,
        num_layers: int = 8,
        d_embedding: int = 8,
        dropout_prob: float = 0.1,
        active: str = "elu",
        norm: str = "batch",
        cosine_latent: bool = False,
        heteroscedastic: bool = False,
    ):
        super().__init__()
        self.heteroscedastic = heteroscedastic

        self.encoder = TabDenseEncoder(
            continuous_cols,
            latent_size,
            growth_rate,
            num_layers,
            d_embedding,
            dropout_prob,
            active,
            norm,
            cosine_latent=cosine_latent,
        )

        self._output_cols = output_cols if output_cols is not None else continuous_cols
        rec_out = 2 * self._output_cols if heteroscedastic else self._output_cols
        self.decoder = TabDenseDecoder(
            latent_size,
            rec_out,
            growth_rate,
            num_layers,
            dropout_prob,
            active,
            norm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        if self.heteroscedastic:
            mean, logvar = out.chunk(2, dim=-1)
            self._last_logvar = logvar
            return mean, encoded
        return out, encoded

# loading the packages
import torch
import torch.nn as nn
 
from rtdl_num_embeddings import (
    PeriodicEmbeddings,
    compute_bins,
)

class ResBlock(nn.Module):
    '''
    Defining an individual residual block as required for ResNets in pytorch
    '''
    def __init__(self, in_features, out_features, dropout_prob=0.1, activ='elu', norm='batch'):
        super(ResBlock, self).__init__()
        self.lin1 = nn.Linear(in_features, out_features, bias=False)
        if norm == 'batch':
            self.normal = nn.BatchNorm1d(out_features)
        elif norm == 'layer':
            self.normal = nn.LayerNorm(out_features)
        else:
            raise ValueError(f"Unsupported norm type: {norm}. Use 'batch' or 'layer'")
        
        if activ == 'elu':
            self.activ = nn.ELU(inplace=True)
        elif activ == 'gelu':
            self.activ = nn.GELU()
        elif activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation type: {activ}. Use 'elu', 'gelu', or 'relu'")
        self.dp = nn.Dropout(p=dropout_prob)
        self.lin2 = nn.Linear(out_features, out_features, bias=False)
        
        # resizing the layers in case of a difference
        if in_features != out_features:
            self.resize = nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                self.normal,
            )
        else:
            self.resize = None

    def forward(self, x):
        identity = x  # for skip-connection
        
        out = self.lin1(x)
        out = self.normal(out)
        out = self.activ(out)
        out = self.dp(out)
        
        out = self.lin2(out)
        out = self.normal(out)
        
        # resize if need be
        if self.resize is not None:
            identity = self.resize(identity)
        
        # skip-connect and finish block
        out += identity  
        out = self.activ(out)
        return out

class DenseResnet(nn.Module):
    '''
    Fabricates the ResNet, for which the size of the blocks changes according to what is passed to the encoder (decoder is symmetric)
    '''
    def __init__(self, input_dim, blocks_dims, num_blocks_per_layer=1, pe=False, d_embedding = 8, activ='elu', norm='batch'):
        super(DenseResnet, self).__init__()

        # creates the network using the blocks defined in the class ResBlock and the input dimensions
        layers = []
        for i, dim in enumerate(blocks_dims):
            if i == 0:
                if pe:
                    layers.append(PeriodicEmbeddings(input_dim, d_embedding=d_embedding, lite=False))
                    layers.append(nn.Flatten())
                    layers.append(nn.Linear(input_dim * d_embedding, dim))
                    layers.append(ResBlock(dim, dim, activ=activ, norm=norm))
                else:
                    layers.append(ResBlock(input_dim, dim, activ=activ, norm=norm))
            else:
                input_dim_for_block = blocks_dims[i-1] if _ == 0 else dim
                for j in range(num_blocks_per_layer):
                    if j == 0:
                        layers.append(ResBlock(blocks_dims[i-1], dim, activ=activ, norm=norm))
                    else:
                        layers.append(ResBlock(dim, dim, activ=activ, norm=norm))
                    
        self.dense_resnet = nn.Sequential(*layers)

    def forward(self, x):
        # forward pass is just going through the blocks, which have their forward pass in the class ResBlock
        return self.dense_resnet(x)

class TabResnetEncoder(nn.Module):
    '''
    Redundant (just calls the class DenseResnet), but matches the shape of the pytorch-widedeep networks for potential model tuning later.
    '''
    def __init__(self, continuous_cols, blocks_dims, pe_bool=True, d_embedding=8, activ='elu', norm='batch'):
        super(TabResnetEncoder, self).__init__()

        input_dim = continuous_cols  # Length of the data, e.g., 153
        self.encoder = DenseResnet(input_dim=input_dim, blocks_dims=blocks_dims, pe=pe_bool, d_embedding=d_embedding, activ=activ, norm=norm)

    def forward(self, x):
        return self.encoder(x)

class TabResnet(nn.Module):
    def __init__(self, continuous_cols, blocks_dims, output_cols=None, d_embedding=8, activ='elu', norm='batch'):

        super(TabResnet, self).__init__()

        self.encoder = TabResnetEncoder(continuous_cols=continuous_cols, blocks_dims=blocks_dims, d_embedding=d_embedding, activ=activ, pe_bool=True, norm=norm)

        # mirrored
        self.decoder = DenseResnet(input_dim=blocks_dims[-1], blocks_dims=blocks_dims[::-1], d_embedding=d_embedding, activ=activ, pe=False, norm=norm)

        # setting output size
        if output_cols is None:
            output_cols = continuous_cols
        # final linear layer for reshaping
        self.reconstruction_layer = nn.Linear(blocks_dims[0], output_cols, bias=False)

    def forward(self, x):
        # Encode the input
        encoded = self.encoder(x)

        # Decode to reconstruct input
        decoded = self.decoder(encoded)
        out = self.reconstruction_layer(decoded)

        # return probe_out, out
        return out, encoded

    def _setup_finetune_optimizer(self, linearprobe, ftopt, ftlr, ftl2, enc_lr, head_lambda, encoder_lambda):
        if linearprobe:
            for p in self.model.parameters():
                p.requires_grad = False
            if ftopt == "adam":
                optimizer = optim.Adam(self.lp.parameters(), lr=ftlr, weight_decay=ftl2)
            elif ftopt == "sgd":
                optimizer = optim.SGD(self.lp.parameters(), lr=ftlr, momentum=0.9, weight_decay=ftl2)
            elif ftopt == "adamw":
                optimizer = optim.AdamW(self.lp.parameters(), lr=ftlr, weight_decay=ftl2)
            else:
                raise ValueError(f"Unknown ftopt {ftopt!r}")
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=head_lambda)
        else:
            if ftopt == "adam":
                optimizer = optim.Adam([
                    {"params": self.model.parameters(), "lr": enc_lr},
                    {"params": self.ft.parameters(), "lr": ftlr, "weight_decay": ftl2},
                ])
            elif ftopt == "sgd":
                optimizer = optim.SGD([
                    {"params": self.model.parameters(), "lr": enc_lr},
                    {"params": self.ft.parameters(), "lr": ftlr, "momentum": 0.9, "weight_decay": ftl2},
                ])
            elif ftopt == "adamw":
                optimizer = optim.AdamW([
                    {"params": self.model.parameters(), "lr": enc_lr},
                    {"params": self.ft.parameters(), "lr": ftlr, "weight_decay": ftl2},
                ])
            else:
                raise ValueError(f"Unknown ftopt {ftopt!r}")
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[encoder_lambda, head_lambda])
        return optimizer, scheduler

    def _setup_finetune_criteria(self, ftlf, rncloss):
        criterion, criterion2, rnc = None, None, None
        if ftlf in ("wmse", "wgnll"):
            criterion = WeightedMaskedMSELoss()
        elif ftlf == "mse":
            criterion = MaskedMSELoss()
        elif ftlf == "mae":
            criterion = MaskedMAELoss()

        if rncloss:
            rnc = RnCLoss(temperature=2, label_diff="l1", feature_sim="l2")

        if ftlf in ("gnll", "wgnll"):
            criterion2 = MaskedGaussianNLLLoss()

        return criterion, criterion2, rnc

    def _apply_batch_masking(self, X_batch, eX_batch, ctx: FinetuneContext):
        if ctx.maskft and ctx.pert_features:
            return self._apply_mask(X_batch + self._pert_noise(X_batch, eX_batch))
        elif ctx.pert_features and not ctx.maskft:
            X_masked = X_batch + self._pert_noise(X_batch, eX_batch)
            mask = torch.zeros_like(X_batch, dtype=torch.bool, device=X_batch.device)
            return X_masked, mask, ~torch.isnan(X_batch)
        elif ctx.maskft and not ctx.pert_features:
            return self._apply_mask(X_batch)
        else:
            mask = torch.zeros_like(X_batch, dtype=torch.bool, device=X_batch.device)
            return X_batch.clone(), mask, ~torch.isnan(X_batch)

    def _forward_pass(self, X_masked, linearprobe):
        encoded = self.model.encoder(X_masked)
        return self.lp(encoded) if linearprobe else self.ft(encoded), encoded

    def _apply_parallax_mask(self, X_masked, parallax_feature_idx):
        parallax_masked = X_masked.clone()
        parallax_masked[:, parallax_feature_idx] = -9999
        indicator_idx = parallax_feature_idx + len(self.feature_cols)
        if indicator_idx < parallax_masked.shape[1]:
            parallax_masked[:, indicator_idx] = 1.0
        return parallax_masked

    def _compute_base_loss(self, y_batch, y_head, batch, ctx: FinetuneContext):
        if ctx.ftlf in ("wmse", "wgnll"):
            return ctx.criterion(y_batch, y_head, 1 / (batch[3] + 1e-5) ** 2)
        elif ctx.ftlf in ("mse", "mae"):
            return ctx.criterion(y_batch, y_head)
        elif ctx.ftlf == "quantile":
            quantiles = torch.tensor([0.16, 0.5, 0.84], device=self.device)
            sw = _sigma_pinball_weights(batch[3], y_batch, ctx.ft_sigma_weight_floor, ctx.ft_sigma_weight_max, ctx.ft_sigma_weight_normalize_batch) if ctx.ft_use_sigma_quantile_weights else None
            return quantile_loss(y_head, y_batch, quantiles, ctx.q_weight_t, sample_weight=sw)
        return 0

    def _compute_parallax_mle(self, y_raw, y_head, X_batch, eX_batch, p_idx, ctx: FinetuneContext):
        pi_gaia = ctx.m_consistency * X_batch[:, self.parallax_feature_idx] + ctx.c_consistency
        sigma_gaia = ctx.m_consistency * eX_batch[:, self.parallax_feature_idx] * ctx.parallax_sigma_scale

        if y_raw.dim() == 3:
            mu_phot = y_head[:, p_idx, 1]
            sigma_phot = 0.5 * (y_head[:, p_idx, 2] - y_head[:, p_idx, 0])
        else:
            mu_phot = y_head[:, p_idx]
            sigma_phot = None

        var = sigma_gaia**2 + (sigma_phot**2 if sigma_phot is not None else 0) + (ctx.parallax_sigma_floor**2 if ctx.parallax_sigma_floor > 0 else 0)

        mle_mask = (~torch.isnan(mu_phot)) & (~torch.isnan(pi_gaia)) & (~torch.isnan(var))
        if mle_mask.any():
            return (((mu_phot - pi_gaia) ** 2) / (var + 1e-8))[mle_mask].mean()
        return 0

    def _apply_parallax_masked_forward(self, X_masked, y_batch, y_raw, ctx: FinetuneContext):
        if ctx.parallax_use_masked_pred and self.parallax_feature_idx is not None:
            p_idx = ctx.parallax_label_idx if ctx.parallax_label_idx is not None else y_batch.shape[1] - 1
            parallax_masked = self._apply_parallax_mask(X_masked, self.parallax_feature_idx)
            y_raw_masked, _ = self._forward_pass(parallax_masked, ctx.linearprobe)
            if y_raw.dim() == 3:
                y_raw[:, p_idx, :] = y_raw_masked[:, p_idx, :]
            else:
                y_raw[:, p_idx] = y_raw_masked[:, p_idx]
        return y_raw

    def _compute_finetune_batch_loss(self, batch, ctx: FinetuneContext):
        X_batch, eX_batch, y_batch, e_y_batch = batch

        X_masked, mask, nanmask = self._apply_batch_masking(X_batch, eX_batch, ctx)

        if ctx.pert_labels:
            y_batch = y_batch + torch.randn_like(y_batch, device=y_batch.device) * e_y_batch

        y_raw, encoded = self._forward_pass(X_masked, ctx.linearprobe)
        y_raw = self._apply_parallax_masked_forward(X_masked, y_batch, y_raw, ctx)

        if ctx.ftlf == "quantile":
            y_head, y_pred_err = y_raw, None
        else:
            y_head, y_pred_err = _reduce_finetune_prediction(y_raw, ctx.ftlf, ctx.linearprobe)

        loss = self._compute_base_loss(y_batch, y_head, batch, ctx)

        if ctx.parallax_mle_weight > 0 and self.parallax_feature_idx is not None and ctx.m_consistency is not None:
            p_idx = ctx.parallax_label_idx if ctx.parallax_label_idx is not None else y_batch.shape[1] - 1
            loss += ctx.parallax_mle_weight * self._compute_parallax_mle(y_raw, y_head, X_batch, eX_batch, p_idx, ctx)

        if ctx.multitask:
            X_reconstructed, _ = self.model(X_masked)
            reconstruction_mask = mask[:, : -self.diff] & nanmask[:, : -self.diff]
            reconstruction_w = 1.0 / (eX_batch[:, : -self.diff] ** 2 + 1e-8)
            rec = self.loss_fn(X_batch[:, : -self.diff], X_reconstructed, reconstruction_mask, reconstruction_w)
            loss = ctx.ft_lambda_pred * loss + ctx.ft_lambda_rec * rec

        if ctx.rncloss:
            try:
                X_m_2, _, _ = self._apply_batch_masking(X_batch, eX_batch, ctx)
                _, encoded_2 = self._forward_pass(X_m_2, False)
                loss += ctx.rnc(torch.stack((encoded, encoded_2), dim=1), y_batch)
            except RuntimeError as e:
                print(e)

        if ctx.ftlf in ("gnll", "wgnll"):
            if y_pred_err is None:
                raise RuntimeError("Gaussian NLL path requires a (mean, logvar) tuple head; not supported for quantile head")
            loss += ctx.criterion2(y_head, y_batch, torch.ones_like(y_pred_err), torch.ones_like(e_y_batch))

        return loss

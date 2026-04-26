import re

with open("src/masked_stellar_autoencoder/models/model.py", "r") as f:
    text = f.read()

helper_code = open("tmp_helpers.py").read()

fit_val_regex = r"    def fit\([\s\S]*?        return val_loss \/ len\(val_loader\)"

new_funcs = helper_code + """
    def _check_linearprobe_compatibility(self, linearprobe, ftlf, multitask, rncloss):
        if linearprobe:
            if ftlf == "quantile":
                raise ValueError("linearprobe requires finetuning lf 'mse' or 'mae', not 'quantile'")
            if ftlf in ("gnll", "wgnll", "wmse"):
                raise ValueError(f"linearprobe does not support loss type {ftlf!r}")
            if multitask:
                raise ValueError("linearprobe with multitask is unsupported")
            if rncloss:
                raise ValueError("linearprobe with rncloss is unsupported")

    def _init_finetune_head(self, linearprobe, ftlabeldim, ftact):
        if ftact == "relu":
            ftactivationfunc = nn.ReLU()
        elif ftact == "elu":
            ftactivationfunc = nn.ELU()
        elif ftact == "gelu":
            ftactivationfunc = nn.GELU()

        self.lp = None
        if linearprobe:
            self.lp = nn.Linear(self.latent_size, ftlabeldim).to(self.device)
            nn.init.xavier_uniform_(self.lp.weight)
            nn.init.zeros_(self.lp.bias)
            self.ft = None
        else:
            self.ft = PredictionHead(self.latent_size, ftlabeldim, ftactivationfunc).to(self.device)

    def _load_finetune_checkpoint(self, ensemblepath, linearprobe):
        try:
            state_dict = torch_load_trusted(ensemblepath, map_location=self.device)
            self.model.load_state_dict(state_dict["autoencoder_state_dict"])
            if not linearprobe:
                self.ft.load_state_dict(state_dict["prediction_head_state_dict"])
            print("loaded checkpoint")
        except Exception:
            if not linearprobe:
                self.ft.apply(self.init_weights_gelu)
            print("restarting fine-tuning")

    def _build_finetune_context(
        self,
        linearprobe, maskft, multitask, ftlf, rncloss, pert_features, pert_labels,
        parallax_use_masked_pred, parallax_label_idx, ft_use_sigma_quantile_weights,
        ft_sigma_weight_floor, ft_sigma_weight_max, ft_sigma_weight_normalize_batch,
        ft_quantile_label_weights, parallax_mle_weight, consistency_params,
        parallax_sigma_scale, parallax_sigma_floor, ft_lambda_pred, ft_lambda_rec,
    ) -> FinetuneContext:
        criterion, criterion2, rnc = self._setup_finetune_criteria(ftlf, rncloss)
        consistency_params = consistency_params or {}
        m_consistency = torch.tensor(consistency_params["m"], device=self.device) if parallax_mle_weight > 0 and "m" in consistency_params else None
        c_consistency = torch.tensor(consistency_params["c"], device=self.device) if parallax_mle_weight > 0 and "c" in consistency_params else None
        q_weight_t = torch.tensor(ft_quantile_label_weights, dtype=torch.float32, device=self.device) if ft_quantile_label_weights is not None else None

        return FinetuneContext(
            linearprobe=linearprobe,
            maskft=maskft,
            multitask=multitask,
            ftlf=ftlf,
            rncloss=rncloss,
            pert_features=pert_features,
            pert_labels=pert_labels,
            parallax_use_masked_pred=parallax_use_masked_pred,
            parallax_label_idx=parallax_label_idx,
            ft_use_sigma_quantile_weights=ft_use_sigma_quantile_weights,
            ft_sigma_weight_floor=ft_sigma_weight_floor,
            ft_sigma_weight_max=ft_sigma_weight_max,
            ft_sigma_weight_normalize_batch=ft_sigma_weight_normalize_batch,
            q_weight_t=q_weight_t,
            criterion=criterion,
            criterion2=criterion2,
            rnc=rnc,
            parallax_mle_weight=parallax_mle_weight,
            m_consistency=m_consistency,
            c_consistency=c_consistency,
            parallax_sigma_scale=parallax_sigma_scale,
            parallax_sigma_floor=parallax_sigma_floor,
            ft_lambda_pred=ft_lambda_pred,
            ft_lambda_rec=ft_lambda_rec,
        )

    def fit(
        self,
        X_train,
        eX_train,
        y_train,
        e_y_train=None,
        X_val=None,
        eX_val=None,
        y_val=None,
        e_y_val=None,
        num_epochs=10,
        mini_batch=32,
        linearprobe=False,
        maskft=False,
        multitask=False,
        rncloss=False,
        last=False,
        ftlr=1e-3,
        ftopt="adam",
        ftact="relu",
        ftl2=0.0,
        ftlf="mse",
        ftdim="1layer512",
        ftlabeldim=5,
        test_stuff=None,
        pt_epoch=0,
        pert_features=False,
        pert_labels=False,
        feature_seed=42,
        ensemblepath=None,
        ft_lambda_pred=0.8,
        ft_lambda_rec=0.2,
        ft_quantile_label_weights: Optional[list] = None,
        ft_use_sigma_quantile_weights: bool = False,
        ft_sigma_weight_floor: float = 1e-6,
        ft_sigma_weight_max: float = 1e6,
        ft_sigma_weight_normalize_batch: bool = True,
        ft_encoder_lr: Optional[float] = None,
        ft_scheduler_encoder_decay: float = 0.95,
        ft_scheduler_head_decay: float = 0.5,
        ft_scheduler_head_step_epochs: int = 10,
        parallax_mle_weight: float = 0.0,
        parallax_use_masked_pred: bool = False,
        parallax_label_idx: Optional[int] = None,
        parallax_sigma_floor: float = 0.0,
        parallax_sigma_scale: float = 1.0,
        consistency_params: Optional[dict] = None,
    ):
        X_train = torch.Tensor(X_train).to(self.device)
        eX_train = torch.Tensor(eX_train).to(self.device)
        y_train = torch.Tensor(y_train).to(self.device)
        e_y_train = torch.Tensor(e_y_train).to(self.device)
        rdataset = TensorDataset(X_train, eX_train, y_train, e_y_train)
        train_loader = DataLoader(rdataset, batch_size=mini_batch, shuffle=True)

        self._check_linearprobe_compatibility(linearprobe, ftlf, multitask, rncloss)
        self._init_finetune_head(linearprobe, ftlabeldim, ftact)
        self._load_finetune_checkpoint(ensemblepath, linearprobe)

        ctx = self._build_finetune_context(
            linearprobe, maskft, multitask, ftlf, rncloss, pert_features, pert_labels,
            parallax_use_masked_pred, parallax_label_idx, ft_use_sigma_quantile_weights,
            ft_sigma_weight_floor, ft_sigma_weight_max, ft_sigma_weight_normalize_batch,
            ft_quantile_label_weights, parallax_mle_weight, consistency_params,
            parallax_sigma_scale, parallax_sigma_floor, ft_lambda_pred, ft_lambda_rec,
        )

        enc_lr = float(ft_encoder_lr) if ft_encoder_lr is not None else float(self.lr)
        head_step = max(1, int(ft_scheduler_head_step_epochs))
        head_lambda = lambda epoch, h=ft_scheduler_head_decay, s=head_step: h ** (epoch // s)
        encoder_lambda = lambda epoch, b=ft_scheduler_encoder_decay: b**epoch

        optimizer, scheduler = self._setup_finetune_optimizer(linearprobe, ftopt, ftlr, ftl2, enc_lr, head_lambda, encoder_lambda)

        os.makedirs(os.path.dirname(self.ft_log_file) if os.path.dirname(self.ft_log_file) else ".", exist_ok=True)
        if _ft_sd := os.path.dirname(self.ft_save_str):
            os.makedirs(_ft_sd, exist_ok=True)
        logging.basicConfig(filename=self.ft_log_file, level=logging.INFO, format="%(asctime)s - Sub-Epoch: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filemode="a", force=True)

        if pert_features or pert_labels:
            random.seed(feature_seed)
            torch.manual_seed(feature_seed)

        for epoch in range(num_epochs):
            if linearprobe:
                self.model.eval()
                self.lp.train()
            else:
                self.model.train()
                self.ft.train()
            epoch_loss = 0

            for batch in train_loader:
                loss = self._compute_finetune_batch_loss(batch, ctx)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.lp.parameters()) if linearprobe else list(self.model.parameters()) + list(self.ft.parameters()), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            print(f"Training Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader)}")
            logging.info(f"Training Loss: {epoch_loss / len(train_loader)}")

            if X_val is not None and y_val is not None:
                validation_loss = self.validate_fit(
                    X_val, eX_val, y_val, e_y_val=e_y_val, mini_batch=mini_batch,
                    linearprobe=linearprobe, maskft=maskft, multitask=multitask,
                    ftlf=ftlf, rncloss=rncloss, ftlabeldim=ftlabeldim,
                    ft_lambda_pred=ft_lambda_pred, ft_lambda_rec=ft_lambda_rec,
                    ft_quantile_label_weights=ft_quantile_label_weights,
                    ft_use_sigma_quantile_weights=ft_use_sigma_quantile_weights,
                    ft_sigma_weight_floor=ft_sigma_weight_floor,
                    ft_sigma_weight_normalize_batch=ft_sigma_weight_normalize_batch,
                    parallax_mle_weight=parallax_mle_weight,
                    parallax_use_masked_pred=parallax_use_masked_pred,
                    parallax_label_idx=parallax_label_idx,
                    parallax_sigma_floor=parallax_sigma_floor,
                    parallax_sigma_scale=parallax_sigma_scale,
                    consistency_params=consistency_params,
                )
                logging.info(f"Validation Loss: {validation_loss}")

            head_sd = self.lp.state_dict() if linearprobe else self.ft.state_dict()
            sd_to_save = {
                "autoencoder_state_dict": self.model.state_dict(),
                "prediction_head_state_dict": head_sd,
                "linear_probe": bool(linearprobe),
                "featurescaler": self.featurescaler,
                "label_scalers": getattr(self, "label_scalers", None),
            }
            torch.save(sd_to_save, self.ft_save_str)
            if self.checkpoint_interval is not None and (epoch + 1) % self.checkpoint_interval == 0:
                torch.save(sd_to_save, self.ft_save_str.split(".")[0] + "_checkpoint_" + str(self.checkpoint_interval) + ".pth")

    def validate_fit(
        self,
        X_val,
        eX_val,
        y_val,
        e_y_val=None,
        mini_batch=32,
        linearprobe=False,
        maskft=False,
        multitask=False,
        ftlf="mse",
        rncloss=False,
        ftlabeldim=5,
        ft_lambda_pred=0.8,
        ft_lambda_rec=0.2,
        ft_quantile_label_weights: Optional[list] = None,
        ft_use_sigma_quantile_weights: bool = False,
        ft_sigma_weight_floor: float = 1e-6,
        ft_sigma_weight_max: float = 1e6,
        ft_sigma_weight_normalize_batch: bool = True,
        parallax_mle_weight: float = 0.0,
        parallax_use_masked_pred: bool = False,
        parallax_label_idx: Optional[int] = None,
        parallax_sigma_floor: float = 0.0,
        parallax_sigma_scale: float = 1.0,
        consistency_params: Optional[dict] = None,
    ):
        self.model.eval()
        if linearprobe:
            self.lp.eval()
        else:
            self.ft.eval()

        val_loss = 0
        X_val, eX_val = torch.Tensor(X_val).to(self.device), torch.Tensor(eX_val).to(self.device)
        y_val, e_y_val = torch.Tensor(y_val).to(self.device), torch.Tensor(e_y_val).to(self.device)
        rdataset = TensorDataset(X_val, eX_val, y_val, e_y_val)
        val_loader = DataLoader(rdataset, batch_size=mini_batch, shuffle=True)

        ctx = self._build_finetune_context(
            linearprobe, maskft, multitask, ftlf, rncloss, False, False,
            parallax_use_masked_pred, parallax_label_idx, ft_use_sigma_quantile_weights,
            ft_sigma_weight_floor, ft_sigma_weight_max, ft_sigma_weight_normalize_batch,
            ft_quantile_label_weights, parallax_mle_weight, consistency_params,
            parallax_sigma_scale, parallax_sigma_floor, ft_lambda_pred, ft_lambda_rec,
        )

        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_finetune_batch_loss(batch, ctx)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")
        return val_loss / len(val_loader)
"""

new_text = re.sub(fit_val_regex, new_funcs.strip("\n"), text)
with open("src/masked_stellar_autoencoder/models/model.py", "w") as f:
    f.write(new_text)

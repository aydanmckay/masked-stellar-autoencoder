"""
Shared fine-tuning data preparation (splits, scaling) for finetune_msa, eval_ensemble, and pilots.
"""

from typing import Any

import numpy as np
import pandas as pd
from astropy.table import Table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from .astrometry_features import (
    apply_parallax_input_policy,
    parallax_label_asinh,
    parallax_label_error_asinh,
    parallax_label_error_log10,
    parallax_label_log10,
)
from .config_paths import expand_config_paths


def _filter_metal_poor(
    data: pd.DataFrame, errordata: pd.DataFrame, mp: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = np.ones(len(data), dtype=bool)
    if mp.get("require_finite_e_fe_h"):
        keep &= data["e_fe_h"].notna().to_numpy()
    if mp.get("max_e_fe_h") is not None:
        mx = float(mp["max_e_fe_h"])
        keep &= data["e_fe_h"].notna().to_numpy() & (data["e_fe_h"].to_numpy() <= mx)
    if mp.get("min_teff") is not None:
        keep &= data["teff"].notna().to_numpy() & (
            data["teff"].to_numpy() >= float(mp["min_teff"])
        )
    if mp.get("max_teff") is not None:
        keep &= data["teff"].notna().to_numpy() & (
            data["teff"].to_numpy() <= float(mp["max_teff"])
        )
    if not keep.all():
        n_drop = int((~keep).sum())
        print(f"metal_poor filters: dropping {n_drop} / {len(data)} rows")
        data = data.iloc[keep].reset_index(drop=True)
        errordata = errordata.iloc[keep].reset_index(drop=True)
    return data, errordata


def _split_data(
    data: pd.DataFrame, errordata: pd.DataFrame, mp: dict[str, Any]
) -> tuple[np.ndarray, ...]:
    err_safe = errordata.fillna(errordata.quantile(0.9))
    errordata = err_safe.fillna(err_safe.median()).fillna(1.0)

    stratify_labels = None
    if mp.get("stratify_feh"):
        fh = data["fe_h"].to_numpy(dtype=float)
        bins = mp.get("feh_stratify_bins", [-np.inf, -2.0, -1.0, 0.0, np.inf])
        strat = pd.cut(fh, bins=bins, labels=False, duplicates="drop")
        stratify_labels = np.asarray(strat, dtype=float)
        stratify_labels[np.isnan(stratify_labels)] = 99
        stratify_labels[np.isnan(fh)] = 99
        stratify_labels = stratify_labels.astype(np.int64)

    try:
        trainset, validset, etrainset, evalidset = train_test_split(
            data.to_numpy(),
            errordata.to_numpy(),
            test_size=0.2,
            random_state=42,
            stratify=stratify_labels,
        )
    except ValueError as e:
        print(f"stratified split failed ({e}); falling back to unstratified split")
        trainset, validset, etrainset, evalidset = train_test_split(
            data.to_numpy(), errordata.to_numpy(), test_size=0.2, random_state=42
        )

    validset, testset, evalidset, etestset = train_test_split(
        validset, evalidset, test_size=0.33, random_state=42
    )
    return trainset, validset, testset, etrainset, evalidset, etestset


def _augment_below_feh(
    trainset: np.ndarray,
    etrainset: np.ndarray,
    mp: dict[str, Any],
    feh_col: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if mp.get("augment_below_feh") is not None and mp.get("augment_fraction", 0) > 0:
        th = float(mp["augment_below_feh"])
        frac = float(mp["augment_fraction"])
        mp_rows = trainset[:, feh_col] < th
        idx_mp = np.flatnonzero(mp_rows)
        if idx_mp.size > 0:
            rng = np.random.default_rng(seed)
            n_add = max(1, int(frac * len(trainset)))
            pick = rng.choice(idx_mp, size=n_add, replace=True)
            trainset = np.vstack([trainset, trainset[pick]])
            etrainset = np.vstack([etrainset, etrainset[pick]])
            print(
                f"metal_poor augment: added {n_add} copies from {idx_mp.size} stars with [Fe/H] < {th}"
            )
    return trainset, etrainset


def _get_scaler_cls(label_scaler_kind: str) -> Any:
    label_scaler_kind = label_scaler_kind.lower()
    if label_scaler_kind == "robust":
        return RobustScaler
    elif label_scaler_kind == "standard":
        return StandardScaler
    elif label_scaler_kind == "power":
        return lambda: PowerTransformer(method="yeo-johnson")
    else:
        raise ValueError(
            f"preprocessing.label_scaler must be 'standard', 'robust', or 'power', got {label_scaler_kind!r}"
        )


def _propagate_label_error(scaler, y, e):
    """Propagate label error through a scaler.

    For linear scalers (StandardScaler, RobustScaler), dividing by scale_ is exact.
    For nonlinear scalers (PowerTransformer), use the delta-method: |PT(y+e) - PT(y)|.
    """
    if isinstance(scaler, PowerTransformer):
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        e_flat = np.asarray(e, dtype=np.float64).ravel()
        return (
            np.abs(
                scaler.transform((y_flat + e_flat).reshape(-1, 1))
                - scaler.transform(y_flat.reshape(-1, 1))
            )
            .ravel()
            .astype(np.float32)
        )
    elif hasattr(scaler, "scale_"):
        return (np.asarray(e, dtype=np.float64) / scaler.scale_.ravel()).astype(
            np.float32
        )
    else:
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        e_flat = np.asarray(e, dtype=np.float64).ravel()
        return (
            np.abs(
                scaler.transform((y_flat + e_flat).reshape(-1, 1))
                - scaler.transform(y_flat.reshape(-1, 1))
            )
            .ravel()
            .astype(np.float32)
        )


def _scale_parallax(
    trainset: np.ndarray,
    validset: np.ndarray,
    etrainset: np.ndarray,
    evalidset: np.ndarray,
    pos: int,
    parallax_target_space: str,
    parallax_floor_mas: float,
    scaler_cls: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    if parallax_target_space == "log10_mas":
        pi_tr = trainset[:, pos].astype(np.float64).copy()
        pi_va = validset[:, pos].astype(np.float64).copy()
        ytr, mtr = parallax_label_log10(pi_tr, parallax_floor_mas)
        yva, mva = parallax_label_log10(pi_va, parallax_floor_mas)
        etr = parallax_label_error_log10(pi_tr, etrainset[:, pos], parallax_floor_mas)
        eva = parallax_label_error_log10(pi_va, evalidset[:, pos], parallax_floor_mas)
    elif parallax_target_space == "asinh_mas":
        pi_tr = trainset[:, pos].astype(np.float64).copy()
        pi_va = validset[:, pos].astype(np.float64).copy()
        ytr, mtr = parallax_label_asinh(pi_tr, scale_mas=1.0)
        yva, mva = parallax_label_asinh(pi_va, scale_mas=1.0)
        etr = parallax_label_error_asinh(pi_tr, etrainset[:, pos], scale_mas=1.0)
        eva = parallax_label_error_asinh(pi_va, evalidset[:, pos], scale_mas=1.0)
    else:
        ytr = trainset[:, pos].astype(np.float64)
        yva = validset[:, pos].astype(np.float64)
        mtr = np.isfinite(ytr)
        mva = np.isfinite(yva)
        etr = etrainset[:, pos].astype(np.float64)
        eva = evalidset[:, pos].astype(np.float64)

    scaler = scaler_cls()
    if np.any(mtr):
        scaler.fit(ytr[mtr].reshape(-1, 1))
    else:
        scaler.fit(np.array([[0.0]], dtype=np.float64))

    label = np.full(len(ytr), np.nan, dtype=np.float32)
    elabel = np.full(len(ytr), np.nan, dtype=np.float32)
    if np.any(mtr):
        label[mtr] = (
            scaler.transform(ytr[mtr].reshape(-1, 1)).astype(np.float32).ravel()
        )
        elabel[mtr] = _propagate_label_error(scaler, ytr[mtr], etr[mtr])

    vlabel = np.full(len(yva), np.nan, dtype=np.float32)
    velabel = np.full(len(yva), np.nan, dtype=np.float32)
    if np.any(mva):
        vlabel[mva] = (
            scaler.transform(yva[mva].reshape(-1, 1)).astype(np.float32).ravel()
        )
        velabel[mva] = _propagate_label_error(scaler, yva[mva], eva[mva])

    return label, elabel, vlabel, velabel, scaler


def _scale_features(
    trainset: np.ndarray,
    validset: np.ndarray,
    testset: np.ndarray,
    etrainset: np.ndarray,
    evalidset: np.ndarray,
    etestset: np.ndarray,
    cols: list,
    pproc_early: dict[str, Any],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler
]:
    featurescaler = RobustScaler()
    featurescaler.fit(trainset)

    if pproc_early.get("xp_feature_scaling", "robust") == "global":
        xp_indices = [
            idx
            for idx, c in enumerate(cols)
            if c.startswith("bp_") or c.startswith("rp_")
        ]
        if xp_indices:
            xp_data = trainset[:, xp_indices]
            q75, q25 = np.nanpercentile(xp_data, [75, 25])
            global_iqr = q75 - q25
            global_median = np.nanmedian(xp_data)
            if global_iqr <= 0:
                global_iqr = 1.0
            featurescaler.center_[xp_indices] = global_median
            featurescaler.scale_[xp_indices] = global_iqr

    trainset = featurescaler.transform(trainset)
    validset = featurescaler.transform(validset)
    testset = featurescaler.transform(testset)
    scale_factors = featurescaler.scale_
    if np.any(scale_factors <= 0):
        scale_factors = np.where(scale_factors <= 0, 1.0, scale_factors)
    etrainset = etrainset / scale_factors
    evalidset = evalidset / scale_factors
    etestset = etestset / scale_factors

    return trainset, validset, testset, etrainset, evalidset, etestset, featurescaler


def _scale_paired_label_blocks(
    target_train: np.ndarray,
    target_valid: np.ndarray,
    num_classes: int,
    scaler_cls: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    scalers = [scaler_cls() for _ in range(int(num_classes / 2))]
    labelled_set = []
    e_labelled_set = []
    vlabelled_set = []
    e_vlabelled_set = []

    for i in range(int(num_classes / 2)):
        y_base = target_train[:, i * 2].reshape(-1, 1)
        labelled_set.append(scalers[i].fit_transform(y_base))
        y_plus = y_base + target_train[:, i * 2 + 1].reshape(-1, 1)
        vy_base = target_valid[:, i * 2].reshape(-1, 1)
        vlabelled_set.append(scalers[i].transform(vy_base))
        vy_plus = vy_base + target_valid[:, i * 2 + 1].reshape(-1, 1)

        # For nonlinear scalers (PowerTransformer), use delta-method for accurate
        # error propagation. For linear scalers (StandardScaler, RobustScaler),
        # dividing by scale_ is exact.
        if isinstance(scalers[i], PowerTransformer):
            elabel = np.abs(
                scalers[i].transform(y_plus) - scalers[i].transform(y_base)
            ).ravel()
            velabel = np.abs(
                scalers[i].transform(vy_plus) - scalers[i].transform(vy_base)
            ).ravel()
        elif hasattr(scalers[i], "scale_"):
            scale_attr = scalers[i].scale_
            elabel = target_train[:, i * 2 + 1] / scale_attr
            velabel = target_valid[:, i * 2 + 1] / scale_attr
        else:
            elabel = np.abs(
                scalers[i].transform(y_plus) - scalers[i].transform(y_base)
            ).ravel()
            velabel = np.abs(
                scalers[i].transform(vy_plus) - scalers[i].transform(vy_base)
            ).ravel()

        e_labelled_set.append(elabel.reshape(-1, 1))
        e_vlabelled_set.append(velabel.reshape(-1, 1))

    labelled_set = np.concatenate(labelled_set, axis=1)
    e_labelled_set = np.concatenate(e_labelled_set, axis=1)
    vlabelled_set = np.concatenate(vlabelled_set, axis=1)
    e_vlabelled_set = np.concatenate(e_vlabelled_set, axis=1)
    return labelled_set, e_labelled_set, vlabelled_set, e_vlabelled_set, scalers


def prepare_finetune_arrays(
    config: dict[str, Any],
    max_train_rows: int | None = None,
    max_valid_rows: int | None = None,
) -> dict[str, Any]:
    """
    Build scaled train/val/test tensors and scalers from finetuning config.

    If max_train_rows / max_valid_rows are set, subsample (first rows) for pilots only.
    """
    expand_config_paths(config)
    data = Table.read(config["data"]["ft_datafile"]).to_pandas()
    errordata = data.copy()

    cols = config["data"]["feature_cols"]
    classes = config["data"]["classes"]
    error_cols = config["data"]["error_cols"]

    data = data[classes + cols]
    errordata = errordata[error_cols]

    mp = config["finetuning"].get("metal_poor") or {}
    data, errordata = _filter_metal_poor(data, errordata, mp)

    trainset, validset, testset, etrainset, evalidset, etestset = _split_data(
        data, errordata, mp
    )

    trainset, etrainset = _augment_below_feh(
        trainset,
        etrainset,
        mp,
        classes.index("fe_h"),
        config["finetuning"].get("seed", 42),
    )

    if max_train_rows is not None and len(trainset) > max_train_rows:
        trainset = trainset[:max_train_rows]
        etrainset = etrainset[:max_train_rows]
    if max_valid_rows is not None and len(validset) > max_valid_rows:
        validset = validset[:max_valid_rows]
        evalidset = evalidset[:max_valid_rows]

    num_classes = len(classes)
    target_train = trainset[:, :num_classes]
    train_feh_raw = target_train[:, classes.index("fe_h")].copy()
    trainset = trainset[:, num_classes:]
    target_valid = validset[:, :num_classes]
    validset = validset[:, num_classes:]
    target_test = testset[:, :num_classes]
    testset = testset[:, num_classes:]

    pproc_early = config.get("preprocessing") or {}
    label_scaler_kind = str(pproc_early.get("label_scaler", "standard")).lower()
    scaler_cls = _get_scaler_cls(label_scaler_kind)

    if pproc_early.get("teff_target_space", "linear") == "log10":
        for ts_arr in [target_train, target_valid, target_test]:
            m_pos = ts_arr[:, 0] > 0
            if np.any(m_pos):
                ts_arr[m_pos, 1] = ts_arr[m_pos, 1] / (ts_arr[m_pos, 0] * np.log(10.0))
                ts_arr[m_pos, 0] = np.log10(ts_arr[m_pos, 0])

    labelled_set, e_labelled_set, vlabelled_set, e_vlabelled_set, scalers = (
        _scale_paired_label_blocks(target_train, target_valid, num_classes, scaler_cls)
    )

    target_set = target_test[:, [i for i in range(num_classes) if i % 2 == 0]]

    # Convert teff in target_set back to physical units so ground truth matches
    # the inverse-transformed predictions in evaluation.
    if pproc_early.get("teff_target_space", "linear") == "log10":
        m_pos = target_set[:, 0] > 0
        if np.any(m_pos):
            target_set[m_pos, 0] = np.power(10.0, target_set[m_pos, 0])

    pos = cols.index("PARALLAX")
    pproc = pproc_early
    astrometry_input_policy = pproc.get("astrometry_input_policy", "legacy_raw")
    astrometry_snr_cap = float(pproc.get("astrometry_snr_cap", 10.0))
    parallax_target_space = pproc.get("parallax_target_space", "linear_mas")
    parallax_floor_mas = float(pproc.get("parallax_floor_mas", 1e-4))

    # Physical Gaia parallax (mas) for test metrics — before input-slot mutation.
    target_parallax_phys = testset[:, pos].astype(np.float64).copy()
    # Formal parallax uncertainty (mas) before input-slot mutation (for ϖ/σ eval bins).
    target_e_parallax_mas = etestset[:, pos].astype(np.float64).copy()
    test_G_mag = (
        testset[:, cols.index("G")].astype(np.float64).copy() if "G" in cols else None
    )
    test_ebv = (
        testset[:, cols.index("EBV")].astype(np.float64).copy()
        if "EBV" in cols
        else None
    )

    label, elabel, vlabel, velabel, scaler = _scale_parallax(
        trainset,
        validset,
        etrainset,
        evalidset,
        pos,
        parallax_target_space,
        parallax_floor_mas,
        scaler_cls,
    )

    apply_parallax_input_policy(
        trainset,
        validset,
        testset,
        etrainset,
        evalidset,
        etestset,
        pos,
        astrometry_input_policy,
        snr_cap=astrometry_snr_cap,
    )

    labelled_set = np.concatenate(
        [labelled_set, np.asarray(label, dtype=np.float32).reshape(-1, 1)], axis=1
    )
    e_labelled_set = np.concatenate(
        [e_labelled_set, np.asarray(elabel, dtype=np.float32).reshape(-1, 1)], axis=1
    )
    scalers.append(scaler)

    target_set = np.concatenate(
        [target_set, target_parallax_phys.reshape(-1, 1)], axis=1
    )
    label_names = ["teff", "logg", "fe_h", "alpha", "age", "parallax"]

    vlabelled_set = np.concatenate(
        [vlabelled_set, np.asarray(vlabel, dtype=np.float32).reshape(-1, 1)], axis=1
    )
    e_vlabelled_set = np.concatenate(
        [e_vlabelled_set, np.asarray(velabel, dtype=np.float32).reshape(-1, 1)], axis=1
    )

    trainset, validset, testset, etrainset, evalidset, etestset, featurescaler = (
        _scale_features(
            trainset,
            validset,
            testset,
            etrainset,
            evalidset,
            etestset,
            cols,
            pproc_early,
        )
    )

    return {
        "trainset": trainset,
        "etrainset": etrainset,
        "labelled_set": labelled_set,
        "e_labelled_set": e_labelled_set,
        "validset": validset,
        "evalidset": evalidset,
        "vlabelled_set": vlabelled_set,
        "e_vlabelled_set": e_vlabelled_set,
        "testset": testset,
        "etestset": etestset,
        "target_set": target_set,
        "scalers": scalers,
        "label_names": label_names,
        "featurescaler": featurescaler,
        "feature_cols": cols,
        "error_cols": error_cols,
        "recon_cols": config["data"]["recon_cols"],
        "train_feh_raw": train_feh_raw,
        "parallax_target_space": parallax_target_space,
        "teff_target_space": pproc_early.get("teff_target_space", "linear"),
        "parallax_floor_mas": parallax_floor_mas,
        "astrometry_input_policy": astrometry_input_policy,
        "label_scaler": label_scaler_kind,
        "target_e_parallax_mas": target_e_parallax_mas,
        "test_G_mag": test_G_mag,
        "test_ebv": test_ebv,
    }

import marimo

__generated_with = "0.1.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    METRICS = "/arc/projects/k-pop/msa_pretrain/metrics.jsonl"
    RESIDUALS = "/arc/projects/k-pop/msa_pretrain/residual_stats.jsonl"
    return METRICS, RESIDUALS, mo, os, pd, plt


@app.cell
def header(mo):
    mo.md("# MSA Pretrain Monitor\nLive metrics from CANFAR staging training run")
    return


@app.cell
def load(mo, os, pd, METRICS):
    refresh = mo.ui.refresh(label="Refresh data", interval="30s")
    if not os.path.exists(METRICS):
        mo.md("Waiting for training to start...")
        return (refresh,)
    df = pd.read_json(METRICS, lines=True)
    return df, refresh


@app.cell
def loss_plot(df, mo, plt):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["epoch"], df["train_loss"], "o-", label="Train", markersize=3)
    if df["val_loss"].notna().any():
        ax.plot(df["epoch"], df["val_loss"], "s-", label="Val", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Reconstruction Loss (MAE)")
    plt.tight_layout()
    mo.md("## Loss Curve")
    return (fig,)


@app.cell
def lr_plot(df, mo, plt):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["epoch"], df["lr"], color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule (Cosine Annealing)")
    plt.tight_layout()
    mo.md("## Learning Rate")
    return (fig,)


@app.cell
def progress(df, mo):
    if len(df) < 1:
        mo.md("## Progress\n_No data yet_")
        return
    elapsed = df["wall_time_s"].sum()
    remaining = int(df["total_epochs"].iloc[-1]) - len(df)
    avg_epoch = elapsed / len(df)
    eta_h = (remaining * avg_epoch) / 3600
    best_val = df["val_loss"].dropna().min() if df["val_loss"].notna().any() else "N/A"
    mo.md(
        f"## Progress\n"
        f"**Epoch {len(df)}/{int(df['total_epochs'].iloc[-1])}** | "
        f"Elapsed: {elapsed / 3600:.1f}h | ETA: {eta_h:.1f}h | "
        f"Best val loss: {best_val}"
    )
    return


@app.cell
def residuals(mo, os, pd, plt, RESIDUALS):
    if not os.path.exists(RESIDUALS):
        mo.md("## Residual Stats\n_No stats yet (appears after epoch 1)_")
        return
    rs = pd.read_json(RESIDUALS, lines=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, col, label in zip(
        axes,
        ["xp_mae", "photo_mae", "overall_mae"],
        ["XP Coeffs", "Photometry", "Overall"],
    ):
        if col in rs.columns:
            ax.plot(rs["epoch"], rs[col], "o-", markersize=3)
        ax.set_title(f"{label} MAE")
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    mo.md("## Residual Stats")
    return (fig,)

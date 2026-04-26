from masked_stellar_autoencoder.training.config_paths import (
    expand_config_paths,
    expand_path,
    ft_checkpoint_paths,
)


def test_expand_path_env(monkeypatch):
    monkeypatch.setenv("SCRATCH", "/tmp/scratch_test")
    assert expand_path("$SCRATCH/msa/x.h5") == "/tmp/scratch_test/msa/x.h5"


def test_expand_config_paths_presaved_null():
    cfg = {"training": {"presaved": None}}
    expand_config_paths(cfg)
    assert cfg["training"]["presaved"] is None


def test_expand_config_paths_presaved_empty_string():
    cfg = {"training": {"presaved": "  "}}
    expand_config_paths(cfg)
    assert cfg["training"]["presaved"] is None


def test_ft_checkpoint_paths_ensemble_suffix():
    cfg = {
        "finetuning": {"ensemble": True},
        "saving": {"model_str": "/out/m.pth", "log_file": "/out/train.log"},
    }
    ms, lg = ft_checkpoint_paths(cfg, 42)
    assert ms == "/out/m_seed42.pth"
    assert lg == "/out/train_seed42.log"


def test_ft_checkpoint_paths_no_ensemble():
    cfg = {
        "finetuning": {"ensemble": False},
        "saving": {"model_str": "/out/m.pth", "log_file": "/out/train.log"},
    }
    assert ft_checkpoint_paths(cfg, 99) == ("/out/m.pth", "/out/train.log")

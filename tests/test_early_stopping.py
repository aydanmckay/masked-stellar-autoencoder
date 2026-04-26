from unittest.mock import MagicMock

import pytest
from models.model import EarlyStopping

torch = pytest.importorskip("torch")


def test_early_stopping_initialization():
    es = EarlyStopping(patience=3, min_delta=0.1, verbose=True, path="test.pth")
    assert es.patience == 3
    assert es.min_delta == 0.1
    assert es.verbose is True
    assert es.path == "test.pth"
    assert es.best_loss is None
    assert es.counter == 0
    assert es.early_stop is False


def test_early_stopping_first_call(mocker):
    mock_save = mocker.patch.object(EarlyStopping, "save_checkpoint")
    es = EarlyStopping()
    model = MagicMock()

    es(1.0, model)

    assert es.best_loss == 1.0
    assert es.counter == 0
    assert es.early_stop is False
    mock_save.assert_called_once_with(model)


def test_early_stopping_improvement(mocker):
    mock_save = mocker.patch.object(EarlyStopping, "save_checkpoint")
    es = EarlyStopping(min_delta=0.1)
    model = MagicMock()

    es(1.0, model)
    mock_save.reset_mock()

    es(0.8, model)

    assert es.best_loss == 0.8
    assert es.counter == 0
    assert es.early_stop is False
    mock_save.assert_called_once_with(model)


def test_early_stopping_no_improvement(mocker):
    mock_save = mocker.patch.object(EarlyStopping, "save_checkpoint")
    es = EarlyStopping(patience=3, min_delta=0.1)
    model = MagicMock()

    es(1.0, model)
    mock_save.reset_mock()

    es(0.95, model)

    assert es.best_loss == 1.0
    assert es.counter == 1
    assert es.early_stop is False
    mock_save.assert_not_called()


def test_early_stopping_trigger(mocker):
    mocker.patch.object(EarlyStopping, "save_checkpoint")
    es = EarlyStopping(patience=2, min_delta=0.0)
    model = MagicMock()

    es(1.0, model)
    es(1.0, model)
    assert es.early_stop is False

    es(1.0, model)
    assert es.early_stop is True


def test_early_stopping_save_checkpoint(mocker):
    mock_save = mocker.patch("models.model.torch.save")
    es = EarlyStopping(path="test_checkpoint.pth")
    model = MagicMock()
    model.state_dict.return_value = {"weight": 1}

    es.save_checkpoint(model)

    mock_save.assert_called_once_with({"weight": 1}, "test_checkpoint.pth")


def test_early_stopping_verbose_output(mocker, capsys):
    mocker.patch.object(EarlyStopping, "save_checkpoint")
    es = EarlyStopping(patience=1, min_delta=0.0, verbose=True)
    model = MagicMock()

    es(1.0, model)
    es(0.5, model)
    captured = capsys.readouterr()
    assert "Validation loss improved to 0.500000, saving model." in captured.out

    es(0.5, model)
    captured = capsys.readouterr()
    assert "EarlyStopping counter: 1 out of 1" in captured.out
    assert "Early stopping triggered." in captured.out

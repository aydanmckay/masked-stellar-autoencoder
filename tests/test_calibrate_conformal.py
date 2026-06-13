import json
from unittest.mock import patch

import numpy as np

from masked_stellar_autoencoder.training.calibrate_conformal import main


def test_calibrate_conformal_cli_writes_json(tmp_path):
    y_val = np.random.default_rng(0).normal(size=(20, 2))
    pred_val = np.stack([y_val - 0.1, y_val, y_val + 0.1], axis=2)
    y_path = tmp_path / "y.npy"
    pred_path = tmp_path / "pred.npy"
    out_path = tmp_path / "calib.json"
    np.save(y_path, y_val)
    np.save(pred_path, pred_val)

    with patch(
        "masked_stellar_autoencoder.training.calibrate_conformal.argparse.ArgumentParser.parse_args",
        return_value=type(
            "Args",
            (),
            {
                "y_val": str(y_path),
                "pred_val": str(pred_path),
                "alpha": 0.1,
                "out": str(out_path),
            },
        )(),
    ):
        main()

    doc = json.loads(out_path.read_text())
    assert doc["method"] == "cqr_asymmetric_quantile_offsets"
    assert len(doc["offsets_lower"]) == 2

import traceback

import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler


def test_power_transformer():
    print("Testing PowerTransformer delta logic...")
    try:
        # Dummy data: 100 rows, 1 target label (e.g. Teff), 1 error col
        target_train = np.random.rand(100, 2) * 1000 + 4000  # Teff
        target_train[:, 1] = target_train[:, 0] * 0.05  # 5% error

        target_valid = np.random.rand(20, 2) * 1000 + 4000
        target_valid[:, 1] = target_valid[:, 0] * 0.05

        scaler = PowerTransformer(method="yeo-johnson")
        y_base = target_train[:, 0].reshape(-1, 1)

        # Simulating finetune_data
        scaler.fit_transform(y_base)
        y_plus = y_base + target_train[:, 1].reshape(-1, 1)

        elabel = np.abs(scaler.transform(y_plus) - scaler.transform(y_base)).ravel()
        print(f"Max transformed error: {np.max(elabel)}")
    except Exception as e:
        print("FAILED power transformer:", e)
        traceback.print_exc()


def test_xp_global_iqr():
    print("Testing XP global IQR logic...")
    try:
        # Dummy XP columns
        cols = ["bp_1", "bp_2", "rp_1", "other", "bp_3"]
        trainset = np.random.randn(100, 5)

        featurescaler = RobustScaler()
        featurescaler.fit(trainset)

        xp_indices = [
            idx
            for idx, c in enumerate(cols)
            if c.startswith("bp_") or c.startswith("rp_")
        ]
        xp_data = trainset[:, xp_indices]
        q75, q25 = np.nanpercentile(xp_data, [75, 25])
        global_iqr = q75 - q25
        global_median = np.nanmedian(xp_data)

        featurescaler.center_[xp_indices] = global_median
        featurescaler.scale_[xp_indices] = global_iqr

        out = featurescaler.transform(trainset)
        print("Success, scaled shape:", out.shape)
    except Exception as e:
        print("FAILED XP IQR:", e)
        traceback.print_exc()


test_power_transformer()
test_xp_global_iqr()

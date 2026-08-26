import numpy as np

from data.data_validator import DataValidator


class TestDataValidator:
    def test_validate_stellar_data_clean(self):
        data = np.array([[10.0, 1.0, 0.5], [15.0, 1.5, 0.2], [20.0, 0.8, 0.1]])
        feature_names = ["G", "BP_RP", "parallax"]
        report = DataValidator.validate_stellar_data(data, feature_names)

        assert report["valid"] is True
        assert len(report["warnings"]) == 0
        assert len(report["errors"]) == 0
        assert report["stats"]["shape"] == (3, 3)
        assert report["stats"]["nan_fraction"] == 0.0
        assert report["stats"]["inf_count"] == 0
        assert report["stats"]["finite_fraction"] == 1.0

    def test_validate_stellar_data_empty(self):
        data = np.array([])
        feature_names = ["G", "BP_RP", "parallax"]
        report = DataValidator.validate_stellar_data(data, feature_names)

        assert report["valid"] is False
        assert len(report["warnings"]) == 0
        assert len(report["errors"]) == 1
        assert "Dataset is empty" in report["errors"][0]

    def test_validate_stellar_data_high_nan(self):
        data = np.array(
            [[10.0, np.nan, np.nan], [15.0, np.nan, np.nan], [20.0, 0.8, 0.1]]
        )
        feature_names = ["G", "BP_RP", "parallax"]
        # 4 NaNs out of 9 values is ~0.44. Let's make it > 0.5
        data = np.array(
            [[np.nan, np.nan, np.nan], [15.0, np.nan, np.nan], [20.0, 0.8, 0.1]]
        )
        # 5 NaNs out of 9 = 0.55 > 0.5
        report = DataValidator.validate_stellar_data(
            data, feature_names, max_nan_fraction=0.5
        )

        assert report["valid"] is True
        assert len(report["warnings"]) == 1
        assert "High NaN fraction" in report["warnings"][0]

    def test_validate_stellar_data_inf_values(self):
        data = np.array([[10.0, np.inf, 0.5], [15.0, 1.5, -np.inf], [20.0, 0.8, 0.1]])
        feature_names = ["G", "BP_RP", "parallax"]
        report = DataValidator.validate_stellar_data(data, feature_names)

        assert report["valid"] is True
        assert len(report["warnings"]) == 1
        assert "Found 2 infinite values" in report["warnings"][0]
        assert report["stats"]["inf_count"] == 2

    def test_validate_stellar_data_g_magnitude_out_of_bounds(self):
        data = np.array(
            [
                [4.0, 1.0, 0.5],  # G < 5
                [26.0, 1.5, 0.2],  # G > 25
                [20.0, 0.8, 0.1],  # Normal
            ]
        )
        feature_names = ["G", "BP_RP", "parallax"]
        report = DataValidator.validate_stellar_data(data, feature_names)

        assert report["valid"] is True
        assert len(report["warnings"]) == 1
        assert "G magnitude values outside typical range" in report["warnings"][0]

    def test_validate_stellar_data_g_magnitude_no_g_feature(self):
        data = np.array(
            [
                [4.0, 1.0, 0.5],  # Out of bounds but G is not in feature_names
                [26.0, 1.5, 0.2],
                [20.0, 0.8, 0.1],
            ]
        )
        feature_names = ["J", "BP_RP", "parallax"]
        report = DataValidator.validate_stellar_data(data, feature_names)

        assert report["valid"] is True
        assert len(report["warnings"]) == 0

    def test_validate_stellar_data_multiple_issues(self):
        data = np.array(
            [[4.0, np.nan, np.inf], [26.0, np.nan, -np.inf], [20.0, np.nan, 0.1]]
        )
        # NaNs = 3/9 = 0.33, max_nan_fraction = 0.2 -> triggers warning
        # Inf = 2 -> triggers warning
        # G bounds -> triggers warning
        feature_names = ["G", "BP_RP", "parallax"]
        report = DataValidator.validate_stellar_data(
            data, feature_names, max_nan_fraction=0.2
        )

        assert report["valid"] is True
        assert len(report["warnings"]) == 3

        warning_texts = [w for w in report["warnings"]]
        assert any("High NaN fraction" in w for w in warning_texts)
        assert any("infinite values" in w for w in warning_texts)
        assert any(
            "G magnitude values outside typical range" in w for w in warning_texts
        )

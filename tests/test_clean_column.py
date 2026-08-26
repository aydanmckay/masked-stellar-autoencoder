import numpy as np
import pytest

from masked_stellar_autoencoder.models.model import TabResnetWrapper


def test_clean_column_byte_strings():
    col_data = np.array([b"1.5", b"-2.3", b"", b"nan", b"NaN"])
    result = TabResnetWrapper._clean_column("test_col", col_data)

    assert np.isnan(result[2])
    # float(b'nan') works in python
    assert np.isnan(result[3])
    assert np.isnan(result[4])
    assert result[0] == 1.5
    assert result[1] == -2.3


def test_clean_column_unicode_strings():
    col_data = np.array(["1.5", "-2.3", "", "nan", "NaN"])
    result = TabResnetWrapper._clean_column("test_col", col_data)

    assert np.isnan(result[2])
    assert np.isnan(result[3])
    assert np.isnan(result[4])
    assert result[0] == 1.5
    assert result[1] == -2.3


def test_clean_column_numeric():
    col_data = np.array([1, 2, 3])
    result = TabResnetWrapper._clean_column("test_col", col_data)

    assert np.array_equal(result, np.array([1, 2, 3]))


def test_clean_column_error():
    col_data = np.array(["abc"])
    with pytest.raises(ValueError):
        TabResnetWrapper._clean_column("test_col", col_data)

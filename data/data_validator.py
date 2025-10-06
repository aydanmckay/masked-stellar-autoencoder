"""
Data validation utilities for the Masked Stellar Autoencoder project.
Provides consistent data quality checks and preprocessing validation.
"""

import numpy as np
import h5py
from typing import List, Dict, Tuple, Optional
import warnings

class DataValidator:
    """Utility class for validating stellar data quality and consistency."""
    
    @staticmethod
    def validate_hdf5_file(filepath: str, required_keys: Optional[List[str]] = None) -> bool:
        """Validate HDF5 file structure and accessibility."""
        try:
            with h5py.File(filepath, 'r') as f:
                if required_keys:
                    missing_keys = [key for key in required_keys if key not in f.keys()]
                    if missing_keys:
                        raise ValueError(f"Missing required keys: {missing_keys}")
                return True
        except Exception as e:
            raise ValueError(f"HDF5 file validation failed: {e}")
    
    @staticmethod
    def validate_stellar_data(data: np.ndarray, 
                            feature_names: List[str],
                            max_nan_fraction: float = 0.5) -> Dict[str, any]:
        """Validate stellar data arrays for quality issues."""
        validation_report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check for completely empty data
        if len(data) == 0:
            validation_report['valid'] = False
            validation_report['errors'].append("Dataset is empty")
            return validation_report
        
        # Check for excessive NaN values
        nan_fraction = np.sum(np.isnan(data)) / data.size
        if nan_fraction > max_nan_fraction:
            validation_report['warnings'].append(f"High NaN fraction: {nan_fraction:.2%}")
        
        # Check for infinite values
        inf_count = np.sum(np.isinf(data))
        if inf_count > 0:
            validation_report['warnings'].append(f"Found {inf_count} infinite values")
        
        # Check for unrealistic stellar parameter ranges
        if 'G' in feature_names:
            g_idx = feature_names.index('G')
            g_values = data[:, g_idx]
            valid_g = g_values[~np.isnan(g_values)]
            if len(valid_g) > 0:
                if np.any(valid_g < 5) or np.any(valid_g > 25):
                    validation_report['warnings'].append("G magnitude values outside typical range [5, 25]")
        
        # Store basic statistics
        validation_report['stats'] = {
            'shape': data.shape,
            'nan_fraction': nan_fraction,
            'inf_count': inf_count,
            'finite_fraction': np.sum(np.isfinite(data)) / data.size
        }
        
        return validation_report
    
    @staticmethod
    def validate_scaling_consistency(scaler, data: np.ndarray) -> bool:
        """Validate that scaler is properly fitted and consistent with data."""
        if not hasattr(scaler, 'scale_'):
            raise ValueError("Scaler not fitted - missing scale_ attribute")
        
        if scaler.scale_ is None:
            raise ValueError("Scaler scale_ attribute is None")
        
        if len(scaler.scale_) != data.shape[1]:
            raise ValueError(f"Scaler dimension mismatch: {len(scaler.scale_)} vs {data.shape[1]}")
        
        if np.any(scaler.scale_ <= 0):
            warnings.warn("Zero or negative scale factors detected")
            return False
        
        return True
    
    @staticmethod
    def check_data_leakage(train_ids: np.ndarray, 
                          val_ids: np.ndarray, 
                          test_ids: Optional[np.ndarray] = None) -> bool:
        """Check for data leakage between train/validation/test sets."""
        # Check train-validation overlap
        train_val_overlap = np.intersect1d(train_ids, val_ids)
        if len(train_val_overlap) > 0:
            raise ValueError(f"Data leakage: {len(train_val_overlap)} samples overlap between train and validation")
        
        if test_ids is not None:
            # Check train-test overlap
            train_test_overlap = np.intersect1d(train_ids, test_ids)
            if len(train_test_overlap) > 0:
                raise ValueError(f"Data leakage: {len(train_test_overlap)} samples overlap between train and test")
            
            # Check validation-test overlap
            val_test_overlap = np.intersect1d(val_ids, test_ids)
            if len(val_test_overlap) > 0:
                raise ValueError(f"Data leakage: {len(val_test_overlap)} samples overlap between validation and test")
        
        return True
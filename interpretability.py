"""
Model Interpretability Module for AquaVista v7.0
===============================================
Provides model interpretability using SHAP and other techniques.
Enhanced with comprehensive debugging and dynamic formatting for all value ranges.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import logging
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from modules.config import Config
from modules.logging_config import get_logger, log_function_call

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
# Try to import LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

class SHAPValidationConfig:
    """Configuration for SHAP validation thresholds"""
    def __init__(self):
        self.allow_unvalidated_shap = True
        self.min_abs_value = 1e-20  # Much more lenient
        self.min_positive_ratio = 0.001  # 0.1% instead of 1%
        self.min_negative_ratio = 0.001  # 0.1% instead of 1%
        self.max_near_zero_ratio = 0.999  # Allow 99.9% near-zero values
        self.enable_adaptive_thresholds = True  # New feature

class ValueRangeAnalyzer:
    """Analyzes value ranges and determines optimal formatting"""
    
    @staticmethod
    def analyze_range(values: np.ndarray) -> Dict[str, Any]:
        """Comprehensive analysis of value range and characteristics"""
        if values is None or len(values) == 0:
            return {"error": "No values provided"}
        
        # Remove any NaN or inf values for analysis
        clean_values = values[np.isfinite(values)]
        if len(clean_values) == 0:
            return {"error": "No finite values found"}
        
        abs_values = np.abs(clean_values)
        non_zero_mask = abs_values > 0
        non_zero_values = abs_values[non_zero_mask]
        
        analysis = {
            "total_count": len(values),
            "finite_count": len(clean_values),
            "zero_count": np.sum(abs_values == 0),
            "non_zero_count": len(non_zero_values),
            "min_value": float(np.min(clean_values)),
            "max_value": float(np.max(clean_values)),
            "min_abs": float(np.min(abs_values)) if len(abs_values) > 0 else 0,
            "max_abs": float(np.max(abs_values)) if len(abs_values) > 0 else 0,
            "mean": float(np.mean(clean_values)),
            "std": float(np.std(clean_values)),
            "median": float(np.median(clean_values)),
            "range_span": float(np.max(abs_values) - np.min(abs_values)) if len(abs_values) > 0 else 0,
        }
        
        # Calculate magnitude characteristics
        if len(non_zero_values) > 0:
            analysis.update({
                "min_non_zero_abs": float(np.min(non_zero_values)),
                "max_non_zero_abs": float(np.max(non_zero_values)),
                "magnitude_range": float(np.max(non_zero_values) / np.min(non_zero_values)),
                "log10_min": float(np.log10(np.min(non_zero_values))),
                "log10_max": float(np.log10(np.max(non_zero_values))),
            })
        else:
            analysis.update({
                "min_non_zero_abs": 0,
                "max_non_zero_abs": 0,
                "magnitude_range": 1,
                "log10_min": 0,
                "log10_max": 0,
            })
        
        return analysis

    @staticmethod
    def get_optimal_format(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal formatting based on comprehensive value analysis"""
        if "error" in analysis:
            return {
                'tick_format': '.3f',
                'text_format': lambda x: f"{x:.3f}",
                'hover_format': '.3f',
                'format_type': 'error_fallback'
            }
        
        max_abs = analysis["max_abs"]
        min_non_zero = analysis.get("min_non_zero_abs", 0)
        magnitude_range = analysis.get("magnitude_range", 1)
        
        # Determine if we need scientific notation
        use_scientific = False
        if max_abs > 0:
            log_max = np.log10(max_abs)
            log_min = np.log10(min_non_zero) if min_non_zero > 0 else -10
            
            # Use scientific notation for very large or very small numbers
            # or when the magnitude range is very large
            if (log_max > 4 or log_min < -3 or (log_max - log_min) > 6):
                use_scientific = True
        
        if use_scientific:
            return {
                'tick_format': '.2e',
                'text_format': lambda x: f"{x:.2e}" if abs(x) > 1e-15 else "0",
                'hover_format': '.3e',
                'format_type': 'scientific',
                'range_info': f"Range: [{analysis['min_value']:.2e}, {analysis['max_value']:.2e}]"
            }
        
        # Fixed-point notation with dynamic precision
        if max_abs >= 1000:
            decimals = 0
        elif max_abs >= 100:
            decimals = 1
        elif max_abs >= 10:
            decimals = 2
        elif max_abs >= 1:
            decimals = 3
        elif max_abs >= 0.1:
            decimals = 4
        elif max_abs >= 0.01:
            decimals = 5
        elif max_abs >= 0.001:
            decimals = 6
        else:
            # Very small numbers - use more decimals
            if min_non_zero > 0:
                needed_decimals = max(6, int(-np.log10(min_non_zero)) + 2)
                decimals = min(needed_decimals, 10)  # Cap at 10 decimals
            else:
                decimals = 6
        
        return {
            'tick_format': f'.{decimals}f',
            'text_format': lambda x: f"{x:.{decimals}f}" if abs(x) > 1e-15 else "0",
            'hover_format': f'.{decimals + 1}f',
            'format_type': 'fixed_point',
            'decimals': decimals,
            'range_info': f"Range: [{analysis['min_value']:.{decimals}f}, {analysis['max_value']:.{decimals}f}]"
        }

class InterpretabilityEngine:
    """Handles model interpretability and explainability"""
    
    def __init__(self, config: Config):
        self.config = config
        self.shap_available = SHAP_AVAILABLE and config.features.shap_analysis
        self.lime_available = LIME_AVAILABLE
        self.shap_validation_config = SHAPValidationConfig()
        self.range_analyzer = ValueRangeAnalyzer()
        
        if not self.shap_available:
            logger.warning("SHAP not available. Install shap for interpretability features.")
    
    def debug_model_and_data(self, model: Any, X_data: pd.DataFrame, y_data: pd.Series = None):
        """Comprehensive model and data debugging with enhanced output"""
        logger.info("=== ENHANCED MODEL AND DATA DEBUGGING ===")
        
        model_type = type(model).__name__
        logger.info(f"Model type: {model_type}")
        
        # Data info
        logger.info(f"X_data shape: {X_data.shape}")
        logger.info(f"X_data dtypes summary: {X_data.dtypes.value_counts().to_dict()}")
        logger.info(f"X_data memory usage: {X_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for problematic values in X_data
        numeric_cols = X_data.select_dtypes(include=[np.number])
        if len(numeric_cols.columns) > 0:
            logger.info(f"Numeric columns stats:")
            for col in numeric_cols.columns[:10]:  # First 10 columns
                col_data = numeric_cols[col]
                logger.info(f"  {col}: range=[{col_data.min():.6f}, {col_data.max():.6f}], "
                           f"std={col_data.std():.6f}, nulls={col_data.isnull().sum()}")
            
            # Check for inf/nan values
            inf_count = np.isinf(numeric_cols.values).sum()
            nan_count = np.isnan(numeric_cols.values).sum()
            logger.info(f"Problematic values: {inf_count} inf, {nan_count} nan")
        
        if y_data is not None:
            logger.info(f"y_data shape: {y_data.shape}")
            logger.info(f"y_data type: {y_data.dtype}")
            
            if pd.api.types.is_numeric_dtype(y_data):
                logger.info(f"y_data range: [{y_data.min():.6f}, {y_data.max():.6f}]")
                logger.info(f"y_data stats: mean={y_data.mean():.6f}, std={y_data.std():.6f}")
                logger.info(f"y_data unique values: {y_data.nunique()}")
            else:
                logger.info(f"y_data unique values: {y_data.value_counts().to_dict()}")
        
        # Enhanced model predictions analysis
        sample_size = min(100, len(X_data))
        X_sample = X_data.head(sample_size)
        
        try:
            predictions = model.predict(X_sample)
            logger.info(f"Prediction analysis on {sample_size} samples:")
            logger.info(f"  Predictions type: {type(predictions)}")
            logger.info(f"  Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
            logger.info(f"  Mean: {np.mean(predictions):.8f}")
            logger.info(f"  Std: {np.std(predictions):.8f}")
            logger.info(f"  Range: [{np.min(predictions):.8f}, {np.max(predictions):.8f}]")
            
            # Detailed prediction distribution
            logger.info(f"  Unique predictions: {len(np.unique(predictions))}")
            logger.info(f"  First 10 predictions: {predictions[:10]}")
            
            # Check prediction variance
            pred_variance = np.var(predictions)
            logger.info(f"  Prediction variance: {pred_variance:.10f}")
            
            if pred_variance < 1e-10:
                logger.error("CRITICAL: Predictions have extremely low variance - model may be broken!")
            elif pred_variance < 1e-6:
                logger.warning("WARNING: Predictions have very low variance - SHAP values will be small")
            
        except Exception as e:
            logger.error(f"Error getting model predictions: {e}")
        
        # Model-specific debugging
        logger.info(f"Model attributes:")
        important_attrs = ['feature_importances_', 'coef_', 'n_features_in_', 'classes_']
        for attr in important_attrs:
            if hasattr(model, attr):
                attr_value = getattr(model, attr)
                if hasattr(attr_value, 'shape'):
                    logger.info(f"  {attr}: shape {attr_value.shape}")
                else:
                    logger.info(f"  {attr}: {type(attr_value)}")
        
        logger.info("=== END ENHANCED MODEL AND DATA DEBUGGING ===")
    
    @log_function_call(log_time=True)
    def calculate_shap_values(self, model: Any, X_data: pd.DataFrame,
                            task_type: str, background_samples: int = 100) -> Optional[np.ndarray]:
        """Calculate SHAP values with FIXED baseline consistency across all explainer types"""
        if not self.shap_available:
            logger.warning("SHAP not available")
            return None
        
        try:
            model_type = type(model).__name__
            logger.info(f"=== SHAP CALCULATION START: {model_type} ===")
            
            # Pre-calculation validation
            if X_data.empty:
                logger.error("Empty X_data provided to SHAP calculation")
                return None
            
            logger.info(f"Input data shape: {X_data.shape}")
            logger.info(f"Task type: {task_type}")
            logger.info(f"Background samples requested: {background_samples}")
            
            # Sample data for SHAP if too large
            if len(X_data) > 1000:
                logger.info(f"Sampling 1000 instances from {len(X_data)} for SHAP calculation")
                X_sample = X_data.sample(n=1000, random_state=42)
            else:
                X_sample = X_data.copy()
            
            logger.info(f"Processing {len(X_sample)} samples for SHAP")
            
            # CRITICAL FIX: Create consistent background sample for ALL explainers
            # This ensures consistent baseline across all explainer types
            background_size = min(background_samples, len(X_data), 100)
            consistent_background = X_data.sample(n=background_size, random_state=42)
            logger.info(f"Created consistent background sample: {consistent_background.shape}")
            
            # Calculate consistent baseline prediction
            baseline_predictions = model.predict(consistent_background)
            consistent_baseline = float(np.mean(baseline_predictions))
            logger.info(f"Consistent baseline value: {consistent_baseline:.8f}")
            
            # Test model predictions first
            try:
                test_predictions = model.predict(X_sample.head(10))
                logger.info(f"Pre-SHAP prediction test successful: {test_predictions[:5]}")
                
                # Analyze prediction characteristics
                pred_analysis = self.range_analyzer.analyze_range(test_predictions)
                logger.info(f"Prediction range analysis: min={pred_analysis.get('min_value', 'N/A'):.8f}, "
                           f"max={pred_analysis.get('max_value', 'N/A'):.8f}, "
                           f"std={pred_analysis.get('std', 'N/A'):.8f}")
                
            except Exception as e:
                logger.error(f"Model prediction test failed: {e}")
                return None
            
            # Initialize explainer variables
            explainer = None
            shap_values = None
            explainer_type_used = None
            
            # FIXED: Tree-based models with consistent baseline handling
            if any(tree_type in model_type for tree_type in 
                   ['Forest', 'Tree', 'XGB', 'LGBM', 'CatBoost', 'Gradient']):
                logger.info(f"Attempting TreeExplainer for {model_type}")
                
                try:
                    # TreeExplainer with model's natural baseline
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    explainer_type_used = "TreeExplainer"
                    
                    # Log TreeExplainer's baseline
                    if hasattr(explainer, 'expected_value'):
                        tree_baseline = explainer.expected_value
                        if isinstance(tree_baseline, (list, np.ndarray)):
                            tree_baseline = tree_baseline[0] if len(tree_baseline) > 0 else 0
                        logger.info(f"TreeExplainer baseline: {tree_baseline:.8f}")
                        logger.info(f"Consistent baseline: {consistent_baseline:.8f}")
                        logger.info(f"Baseline difference: {abs(tree_baseline - consistent_baseline):.8f}")
                    
                    logger.info("TreeExplainer calculation successful")
                    
                except Exception as e:
                    logger.warning(f"TreeExplainer failed: {e}, trying KernelExplainer with consistent baseline")
                    try:
                        # Fallback: KernelExplainer with our consistent background
                        explainer = shap.KernelExplainer(model.predict, consistent_background)
                        shap_values = explainer.shap_values(X_sample, nsamples=100)
                        explainer_type_used = "KernelExplainer (TreeExplainer fallback)"
                        logger.info("KernelExplainer fallback successful")
                    except Exception as e2:
                        logger.error(f"KernelExplainer fallback also failed: {e2}")
                        return None
            
            # FIXED: Linear models with consistent baseline handling
            elif any(linear_type in model_type for linear_type in 
                    ['Linear', 'Logistic', 'Ridge', 'Lasso', 'ElasticNet']):
                logger.info(f"Attempting LinearExplainer for {model_type}")
                
                try:
                    # LinearExplainer - but we'll verify baseline consistency
                    explainer = shap.LinearExplainer(model, consistent_background)
                    shap_values = explainer.shap_values(X_sample)
                    explainer_type_used = "LinearExplainer"
                    
                    # Log LinearExplainer's baseline and compare
                    if hasattr(explainer, 'expected_value'):
                        linear_baseline = explainer.expected_value
                        if isinstance(linear_baseline, (list, np.ndarray)):
                            linear_baseline = linear_baseline[0] if len(linear_baseline) > 0 else 0
                        logger.info(f"LinearExplainer baseline: {linear_baseline:.8f}")
                        logger.info(f"Consistent baseline: {consistent_baseline:.8f}")
                        logger.info(f"Baseline difference: {abs(linear_baseline - consistent_baseline):.8f}")
                    
                    logger.info("LinearExplainer successful")
                    
                except Exception as e:
                    logger.warning(f"LinearExplainer failed: {e}, using KernelExplainer with consistent baseline")
                    try:
                        # Fallback: KernelExplainer with our consistent background
                        explainer = shap.KernelExplainer(model.predict, consistent_background)
                        shap_values = explainer.shap_values(X_sample, nsamples=100)
                        explainer_type_used = "KernelExplainer (LinearExplainer fallback)"
                        logger.info("KernelExplainer fallback successful")
                    except Exception as e2:
                        logger.error(f"KernelExplainer fallback failed: {e2}")
                        return None
            
            # FIXED: All other models - use KernelExplainer with consistent baseline
            else:
                logger.info(f"Using KernelExplainer for {model_type}")
                try:
                    # Always use our consistent background for KernelExplainer
                    if task_type == 'classification' and hasattr(model, 'predict_proba'):
                        logger.info("Using predict_proba for classification")
                        explainer = shap.KernelExplainer(model.predict_proba, consistent_background)
                    else:
                        logger.info("Using predict for regression/classification")
                        explainer = shap.KernelExplainer(model.predict, consistent_background)
                    
                    shap_values = explainer.shap_values(X_sample, nsamples=100)
                    explainer_type_used = "KernelExplainer"
                    logger.info("KernelExplainer successful")
                    
                except Exception as e:
                    logger.error(f"KernelExplainer failed: {e}")
                    return None
            
            # FIXED: Consistent baseline storage for all explainer types
            if hasattr(explainer, 'expected_value'):
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    stored_baseline = explainer.expected_value[0] if len(explainer.expected_value) > 0 else consistent_baseline
                else:
                    stored_baseline = explainer.expected_value
            else:
                stored_baseline = consistent_baseline
            
            # Always store the baseline for later use
            self.shap_base_value = float(stored_baseline)
            logger.info(f"Stored baseline value: {self.shap_base_value:.8f}")
            logger.info(f"Explainer type used: {explainer_type_used}")
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list) and task_type == 'classification':
                logger.info(f"Multiclass SHAP values - {len(shap_values)} classes")
                for i, class_shap in enumerate(shap_values):
                    logger.info(f"Class {i} shape: {class_shap.shape}")
                    class_analysis = self.range_analyzer.analyze_range(class_shap.flatten())
                    logger.info(f"Class {i} range: [{class_analysis.get('min_value', 0):.8f}, "
                               f"{class_analysis.get('max_value', 0):.8f}]")
                
                # Average across classes for display
                shap_values = np.array(shap_values).mean(axis=0)
                logger.info(f"After averaging across classes: {shap_values.shape}")
            
            # Handle 3D arrays
            if hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                logger.info(f"3D SHAP values detected: {shap_values.shape}")
                if shap_values.shape[2] == 2:
                    shap_values = shap_values[:, :, 1]  # Take positive class
                else:
                    shap_values = shap_values.mean(axis=2)
                logger.info(f"After 3D processing: {shap_values.shape}")
            
            # ENHANCED: Baseline consistency validation
            if shap_values is not None and hasattr(shap_values, 'shape'):
                logger.info(f"=== BASELINE CONSISTENCY VALIDATION ===")
                
                # Test SHAP additivity with our stored baseline
                if len(X_sample) > 0:
                    test_sample = X_sample.iloc[0:1]
                    actual_pred = model.predict(test_sample)[0]
                    
                    if shap_values.ndim == 2:
                        shap_sum = shap_values[0].sum()
                    else:
                        shap_sum = shap_values.sum()
                    
                    expected_pred = self.shap_base_value + shap_sum
                    additivity_error = abs(actual_pred - expected_pred)
                    
                    logger.info(f"Additivity validation:")
                    logger.info(f"  Actual prediction: {actual_pred:.8f}")
                    logger.info(f"  SHAP baseline: {self.shap_base_value:.8f}")
                    logger.info(f"  SHAP sum: {shap_sum:.8f}")
                    logger.info(f"  Expected (baseline + SHAP): {expected_pred:.8f}")
                    logger.info(f"  Additivity error: {additivity_error:.8f}")
                    
                    if additivity_error > 1e-3:  # Tolerance
                        logger.warning(f"Large additivity error detected: {additivity_error:.8f}")
                        logger.warning("SHAP values may not be correctly calibrated")
                    else:
                        logger.info("SHAP additivity validation passed")
                
                # Comprehensive SHAP analysis
                logger.info(f"=== COMPREHENSIVE SHAP ANALYSIS ===")
                
                shap_analysis = self.range_analyzer.analyze_range(shap_values.flatten())
                
                logger.info(f"SHAP Values Shape: {shap_values.shape}")
                logger.info(f"Data Type: {shap_values.dtype}")
                
                # Log all the detailed statistics
                for key, value in shap_analysis.items():
                    if isinstance(value, float):
                        logger.info(f"{key}: {value:.12f}")
                    else:
                        logger.info(f"{key}: {value}")
                
                # Feature-wise analysis (top 10 features)
                if shap_values.ndim == 2:
                    feature_importance = np.abs(shap_values).mean(axis=0)
                    top_features = np.argsort(feature_importance)[::-1][:10]
                    
                    logger.info(f"Top 10 features by mean |SHAP|:")
                    for i, feature_idx in enumerate(top_features):
                        if feature_idx < len(X_sample.columns):
                            feature_name = X_sample.columns[feature_idx]
                            importance = feature_importance[feature_idx]
                            feature_shap = shap_values[:, feature_idx]
                            logger.info(f"  {i+1}. {feature_name}: mean_abs={importance:.12f}, "
                                       f"range=[{feature_shap.min():.8f}, {feature_shap.max():.8f}]")
                
                # Sample-wise analysis (first 5 samples)
                if shap_values.ndim == 2:
                    logger.info(f"Sample-wise SHAP sums (first 5 samples):")
                    for i in range(min(5, shap_values.shape[0])):
                        sample_sum = shap_values[i].sum()
                        sample_abs_sum = np.abs(shap_values[i]).sum()
                        logger.info(f"  Sample {i}: sum={sample_sum:.8f}, abs_sum={sample_abs_sum:.8f}")
                
                # Quality checks
                problematic_indicators = []
                if shap_analysis.get('max_abs', 0) < 1e-12:
                    problematic_indicators.append("All values extremely close to zero")
                
                if shap_analysis.get('std', 0) < 1e-15:
                    problematic_indicators.append("Extremely low variance")
                
                zero_ratio = shap_analysis.get('zero_count', 0) / shap_analysis.get('total_count', 1)
                if zero_ratio > 0.95:
                    problematic_indicators.append(f"Very high zero ratio: {zero_ratio:.1%}")
                
                if problematic_indicators:
                    logger.warning("POTENTIAL ISSUES:")
                    for indicator in problematic_indicators:
                        logger.warning(f"  - {indicator}")
                else:
                    logger.info("SHAP values appear to be valid (non-zero with reasonable range)")
                
                logger.info(f"=== SHAP ANALYSIS COMPLETE ===")
                
            return shap_values
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values for {model_type}: {str(e)}")
            logger.error(f"Full traceback:", exc_info=True)
            return None
    
    def validate_shap_values(self, shap_values: np.ndarray, model_name: str, 
                            min_abs_value: float = None,
                            min_positive_ratio: float = None,
                            min_negative_ratio: float = None) -> Tuple[bool, Dict[str, Any]]:
        """Enhanced SHAP validation with detailed diagnostics"""
        # Use config values if not overridden
        if min_abs_value is None:
            min_abs_value = self.shap_validation_config.min_abs_value
        if min_positive_ratio is None:
            min_positive_ratio = self.shap_validation_config.min_positive_ratio
        if min_negative_ratio is None:
            min_negative_ratio = self.shap_validation_config.min_negative_ratio
        
        if shap_values is None or not isinstance(shap_values, np.ndarray):
            return False, {'reason': 'No SHAP values or invalid type'}
        
        # Use enhanced range analysis
        analysis = self.range_analyzer.analyze_range(shap_values.flatten())
        
        # Enhanced diagnostics
        diagnostics = {
            'model_name': model_name,
            'shape': shap_values.shape,
            'range_analysis': analysis,
            'max_abs_value': analysis.get('max_abs', 0),
            'mean_abs_value': analysis.get('mean', 0),
            'positive_ratio': float(np.sum(shap_values > 0) / shap_values.size),
            'negative_ratio': float(np.sum(shap_values < 0) / shap_values.size),
            'zero_ratio': float(analysis.get('zero_count', 0) / analysis.get('total_count', 1)),
            'has_nan': analysis.get('total_count', 0) != analysis.get('finite_count', 0),
            'has_inf': bool(np.any(np.isinf(shap_values))),
            'std_dev': analysis.get('std', 0),
            'magnitude_range': analysis.get('magnitude_range', 1),
        }
        
        # Enhanced validation with more lenient thresholds for small values
        reasons = []
        
        # Check 1: No NaN or Inf values
        if diagnostics['has_nan'] or diagnostics['has_inf']:
            reasons.append('Contains NaN or Inf values')
        
        # Check 2: Values are not all effectively zero (more lenient threshold)
        if diagnostics['max_abs_value'] < min_abs_value:
            reasons.append(f'All values near zero (max abs: {diagnostics["max_abs_value"]:.2e})')
        
        # Check 3: Has reasonable distribution (more lenient for small values)
        if diagnostics['max_abs_value'] > 1e-10:  # Only check distribution if values aren't extremely small
            if diagnostics['positive_ratio'] < min_positive_ratio:
                reasons.append(f'Too few positive values ({diagnostics["positive_ratio"]*100:.1f}%)')
            
            if diagnostics['negative_ratio'] < min_negative_ratio:
                reasons.append(f'Too few negative values ({diagnostics["negative_ratio"]*100:.1f}%)')
        
        # Check 4: Reasonable variance
        if diagnostics['std_dev'] < 1e-15:
            reasons.append('Extremely low variance in SHAP values')
        
        # Determine if valid
        is_valid = len(reasons) == 0
        diagnostics['is_valid'] = is_valid
        diagnostics['validation_reasons'] = reasons
        
        # Add suggestions for improvement
        suggestions = []
        if not is_valid:
            if 'zero' in str(reasons):
                suggestions.append("Try a different explainer type (Tree vs Kernel vs Linear)")
                suggestions.append("Check if model predictions have sufficient variance")
                suggestions.append("Verify model is properly trained")
            
            if 'positive' in str(reasons) or 'negative' in str(reasons):
                suggestions.append("Model may be producing one-sided predictions")
                suggestions.append("Check target variable distribution")
        
        diagnostics['improvement_suggestions'] = suggestions
        
        return is_valid, diagnostics

    def calculate_permutation_importance(self, model: Any, X_data: pd.DataFrame,
                                       y_data: pd.Series, n_repeats: int = 10) -> Dict[str, Any]:
        """Calculate permutation feature importance"""
        try:
            # Limit data size for performance
            if len(X_data) > 5000:
                sample_idx = np.random.choice(len(X_data), 5000, replace=False)
                X_sample = X_data.iloc[sample_idx]
                y_sample = y_data.iloc[sample_idx]
            else:
                X_sample = X_data
                y_sample = y_data
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_sample, y_sample,
                n_repeats=n_repeats,
                random_state=self.config.computation.random_state,
                n_jobs=self.config.computation.n_jobs
            )
            
            # Create importance dictionary
            importance_dict = {}
            for i, feature in enumerate(X_data.columns):
                importance_dict[feature] = {
                    'importance_mean': perm_importance.importances_mean[i],
                    'importance_std': perm_importance.importances_std[i]
                }
            
            logger.info("Permutation importance calculated successfully")
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {str(e)}")
            return {}
    
    def create_shap_summary_plot(self, shap_values: np.ndarray, feature_names: List[str]) -> go.Figure:
        """Create SHAP summary plot with robust error handling"""
        if shap_values is None:
            return self._create_error_figure("SHAP values not available")

        try:
            # Ensure feature names are valid
            if len(feature_names) != shap_values.shape[1]:
                feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]
            
            # Clean feature names
            clean_names = []
            for name in feature_names:
                clean_name = str(name).strip() if name else f"Feature_{len(clean_names)}"
                if clean_name == '' or clean_name == '0':
                    clean_name = f"Feature_{len(clean_names)}"
                clean_names.append(clean_name)
            
            # Calculate importance
            mean_abs = np.abs(shap_values).mean(axis=0)
            mean_signed = shap_values.mean(axis=0)
            
            # Sort and limit to top features
            sorted_idx = np.argsort(mean_abs)[-20:][::-1]
            top_features = [clean_names[i] for i in sorted_idx]
            top_values = mean_signed[sorted_idx]
            
            # Handle very small values
            max_abs = np.max(np.abs(top_values))
            if max_abs < 1e-15:
                # Use scientific notation for very small values
                text_format = lambda x: f"{x:.2e}" if abs(x) > 1e-20 else "0"
                tick_format = ".2e"
            elif max_abs < 1e-3:
                # Use more decimal places for small values
                decimals = max(6, int(-np.log10(max_abs)) + 2)
                text_format = lambda x: f"{x:.{decimals}f}"
                tick_format = f".{decimals}f"
            else:
                # Standard formatting
                text_format = lambda x: f"{x:.4f}"
                tick_format = ".4f"
            
            # Create diverging bar chart
            fig = go.Figure()
            
            # Split positive and negative
            pos_vals = np.where(top_values > 0, top_values, 0)
            neg_vals = np.where(top_values < 0, top_values, 0)
            
            # Add bars
            fig.add_trace(go.Bar(
                x=neg_vals, y=top_features, orientation="h",
                name="Negative Impact", marker_color="#d32f2f",
                text=[text_format(v) if v != 0 else "" for v in neg_vals],
                textposition="outside"
            ))
            
            fig.add_trace(go.Bar(
                x=pos_vals, y=top_features, orientation="h",
                name="Positive Impact", marker_color="#388e3c",
                text=[f"+{text_format(v)}" if v > 0 else "" for v in pos_vals],
                textposition="outside"
            ))
            
            # Update layout
            fig.update_layout(
                barmode="relative",
                title="SHAP Feature Importance",
                xaxis_title="Mean SHAP Value",
                yaxis_title="Features",
                height=max(500, len(top_features) * 30),
                template="plotly_white"
            )
            
            fig.update_xaxes(tickformat=tick_format)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP plot: {e}")
            return self._create_error_figure(f"Error creating SHAP plot: {str(e)}")

    def _create_error_figure(self, message: str) -> go.Figure:
        """Create figure showing error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="SHAP Analysis Error",
            height=400,
            template="plotly_white"
        )
        return fig
    
    def get_robust_feature_importance(self, model, model_name: str, X_test: pd.DataFrame,
                                    y_test: pd.Series, mc_analysis: Dict = None) -> Dict[str, Any]:
        """Get feature importance that's robust to multicollinearity"""
        
        if mc_analysis and mc_analysis['severity'] == 'high':
            # Use permutation importance instead of SHAP for high multicollinearity
            from sklearn.inspection import permutation_importance
            
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42
            )
            
            return {
                'method': 'permutation',
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std,
                'feature_names': X_test.columns.tolist(),
                'warning': 'Using permutation importance due to high multicollinearity'
            }
        else:
            # Use standard SHAP for low/moderate multicollinearity
            return self.calculate_shap_values(model, X_test, task_type)


    def create_shap_waterfall(self, shap_values: np.ndarray, feature_names: List[str],
                            feature_values: pd.Series) -> go.Figure:
        """Create SHAP waterfall plot for single prediction with enhanced formatting"""
        if shap_values is None:
            fig = go.Figure()
            fig.add_annotation(
                text="SHAP values not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Sort features by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:15]  # Top 15 features
        
        # Analyze value range for formatting
        selected_shap = shap_values[sorted_idx]
        range_analysis = self.range_analyzer.analyze_range(selected_shap)
        format_config = self.range_analyzer.get_optimal_format(range_analysis)
        
        # Prepare data
        features = []
        values = []
        colors = []
        
        for idx in sorted_idx:
            feature = feature_names[idx]
            value = shap_values[idx]
            feature_val = feature_values.iloc[idx]
            
            # Create label with formatted values
            if isinstance(feature_val, (int, float)):
                label = f"{feature} = {feature_val:.3f}"
            else:
                label = f"{feature} = {feature_val}"
            
            features.append(label)
            values.append(value)
            colors.append('#d32f2f' if value < 0 else '#388e3c')  # Red for negative, green for positive
        
        # Add remaining features as "other" if significant
        other_value = sum(shap_values[i] for i in range(len(shap_values)) 
                         if i not in sorted_idx)
        if abs(other_value) > range_analysis.get('max_abs', 0) * 0.01:  # If other > 1% of max
            features.append("Other features")
            values.append(other_value)
            colors.append('#d32f2f' if other_value < 0 else '#388e3c')
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Calculate cumulative values for positioning
        cumulative = [0]
        for val in values:
            cumulative.append(cumulative[-1] + val)
        
        # Add bars with enhanced formatting
        for i, (feature, value, color) in enumerate(zip(features, values, colors)):
            fig.add_trace(go.Bar(
                x=[feature],
                y=[abs(value)],
                base=min(cumulative[i], cumulative[i+1]),
                marker_color=color,
                name='Positive Impact' if color == '#388e3c' else 'Negative Impact',
                showlegend=i == 0 or (i == 1 and colors[0] == colors[1]),  # Show legend for first of each type
                text=format_config['text_format'](value),
                textposition='outside',
                hovertemplate=f'Feature: %{{x}}<br>SHAP: {format_config["text_format"](value)}<extra></extra>'
            ))
        
        # Add base value line if available
        if hasattr(self, 'shap_base_value') and self.shap_base_value is not None:
            fig.add_hline(
                y=self.shap_base_value, 
                line_dash="dash", 
                line_color="gray",
                annotation_text=f"Base value: {format_config['text_format'](self.shap_base_value)}"
            )
        
        # Add prediction line
        total_effect = sum(values)
        prediction = (self.shap_base_value if hasattr(self, 'shap_base_value') else 0) + total_effect
        fig.add_hline(
            y=prediction,
            line_dash="solid",
            line_color="blue",
            annotation_text=f"Prediction: {format_config['text_format'](prediction)}"
        )
        
        fig.update_layout(
            title=f"SHAP Waterfall Plot - Individual Prediction Explanation<br>"
                  f"<sub>Format: {format_config.get('format_type', 'auto')}</sub>",
            xaxis_title="Features",
            yaxis_title="SHAP Value Contribution",
            template='plotly_white',
            height=600,
            showlegend=True,
            xaxis_tickangle=-45,
            yaxis_tickformat=format_config['tick_format']
        )
        
        return fig
    
    def create_shap_interaction_plot(self, shap_values: np.ndarray,
                                   feature_names: List[str],
                                   feature1: str, feature2: str) -> go.Figure:
        """Create SHAP interaction plot between two features with enhanced formatting"""
        if shap_values is None:
            fig = go.Figure()
            fig.add_annotation(
                text="SHAP values not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get feature indices
        try:
            idx1 = feature_names.index(feature1)
            idx2 = feature_names.index(feature2)
        except ValueError:
            fig = go.Figure()
            fig.add_annotation(
                text="Feature not found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get SHAP values for the features
        shap1 = shap_values[:, idx1]
        shap2 = shap_values[:, idx2]
        
        # Analyze ranges for formatting
        combined_values = np.concatenate([shap1, shap2])
        range_analysis = self.range_analyzer.analyze_range(combined_values)
        format_config = self.range_analyzer.get_optimal_format(range_analysis)
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=shap1,
            y=shap2,
            mode='markers',
            marker=dict(
                size=6,
                color=np.abs(shap1 + shap2),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Combined Impact",
                    tickformat=format_config['tick_format']
                )
            ),
            text=[f"Sample {i}" for i in range(len(shap1))],
            hovertemplate=(
                f"{feature1} SHAP: %{{x:{format_config['hover_format']}}}<br>"
                f"{feature2} SHAP: %{{y:{format_config['hover_format']}}}<br>"
                f"Combined: %{{marker.color:{format_config['hover_format']}}}<extra></extra>"
            )
        ))
        
        # Add trend line if correlation exists
        correlation = np.corrcoef(shap1, shap2)[0, 1]
        if abs(correlation) > 0.1:  # Only show trend if meaningful correlation
            z = np.polyfit(shap1, shap2, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(shap1.min(), shap1.max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name=f'Trend (r={correlation:.3f})',
                hovertemplate=f'Trend line (correlation: {correlation:.3f})<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"SHAP Feature Interaction: {feature1} vs {feature2}<br>"
                  f"<sub>Format: {format_config.get('format_type', 'auto')} | "
                  f"Correlation: {correlation:.3f}</sub>",
            xaxis_title=f"{feature1} SHAP Value",
            yaxis_title=f"{feature2} SHAP Value",
            template='plotly_white',
            height=500
        )
        
        # Apply formatting to axes
        fig.update_xaxes(tickformat=format_config['tick_format'])
        fig.update_yaxes(tickformat=format_config['tick_format'])
        
        return fig
    
    def create_feature_dependence_plot(self, shap_values: np.ndarray,
                                     X_data: pd.DataFrame,
                                     feature_name: str) -> go.Figure:
        """Create SHAP dependence plot for a feature with enhanced formatting"""
        if shap_values is None or feature_name not in X_data.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Data not available for dependence plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get feature index
        feature_idx = X_data.columns.tolist().index(feature_name)
        
        # Get feature values and SHAP values
        feature_values = X_data[feature_name].values
        shap_feature = shap_values[:, feature_idx]
        
        # Analyze SHAP range for formatting
        range_analysis = self.range_analyzer.analyze_range(shap_feature)
        format_config = self.range_analyzer.get_optimal_format(range_analysis)
        
        # Find best interaction feature (highest correlation with SHAP values)
        correlations = []
        for i, col in enumerate(X_data.columns):
            if col != feature_name:
                # Handle potential NaN values in correlation
                mask = np.isfinite(shap_feature) & np.isfinite(X_data[col])
                if mask.sum() > 10:  # Need at least 10 valid points
                    corr = np.corrcoef(shap_feature[mask], X_data[col][mask])[0, 1]
                    if np.isfinite(corr):
                        correlations.append((i, col, abs(corr)))
        
        # Select interaction feature
        if correlations:
            best_interaction = max(correlations, key=lambda x: x[2])
            interaction_idx, interaction_name, corr_strength = best_interaction
            interaction_values = X_data[interaction_name].values
            logger.info(f"Selected interaction feature: {interaction_name} (correlation: {corr_strength:.3f})")
        else:
            interaction_values = None
            interaction_name = None
        
        # Create plot
        fig = go.Figure()
        
        # Main scatter plot
        if interaction_values is not None:
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=shap_feature,
                mode='markers',
                marker=dict(
                    size=5,
                    color=interaction_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=interaction_name,
                        x=1.05
                    )
                ),
                name='Data points',
                hovertemplate=(
                    f"{feature_name}: %{{x:.3f}}<br>"
                    f"SHAP: %{{y:{format_config['hover_format']}}}<br>"
                    f"{interaction_name}: %{{marker.color:.3f}}<extra></extra>"
                )
            ))
        else:
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=shap_feature,
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.6),
                name='Data points',
                hovertemplate=(
                    f"{feature_name}: %{{x:.3f}}<br>"
                    f"SHAP: %{{y:{format_config['hover_format']}}}<extra></extra>"
                )
            ))
        
        # Add smoothed trend line if we have enough points
        if len(feature_values) > 20:
            try:
                # Use local polynomial regression for smoothing
                from scipy.interpolate import UnivariateSpline
                
                # Sort data for spline
                sorted_idx = np.argsort(feature_values)
                x_sorted = feature_values[sorted_idx]
                y_sorted = shap_feature[sorted_idx]
                
                # Remove duplicates and NaN values
                valid_mask = np.isfinite(x_sorted) & np.isfinite(y_sorted)
                x_clean = x_sorted[valid_mask]
                y_clean = y_sorted[valid_mask]
                
                if len(x_clean) > 10:
                    # Create spline with smoothing
                    spl = UnivariateSpline(x_clean, y_clean, s=len(x_clean) * np.var(y_clean) * 0.1)
                    x_smooth = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_smooth = spl(x_smooth)
                    
                    fig.add_trace(go.Scatter(
                        x=x_smooth,
                        y=y_smooth,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Trend',
                        hovertemplate=f'Smoothed trend<extra></extra>'
                    ))
            except ImportError:
                logger.info("scipy not available for trend line")
            except Exception as e:
                logger.warning(f"Could not create trend line: {e}")
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Enhanced layout with formatting info
        title_text = f"SHAP Dependence: {feature_name}"
        if interaction_name:
            title_text += f" (colored by {interaction_name})"
        title_text += f"<br><sub>Format: {format_config.get('format_type', 'auto')}</sub>"
        
        fig.update_layout(
            title=title_text,
            xaxis_title=feature_name,
            yaxis_title="SHAP Value",
            template='plotly_white',
            height=500,
            margin=dict(r=150) if interaction_values is not None else dict()
        )
        
        # Apply formatting
        fig.update_yaxes(tickformat=format_config['tick_format'])
        
        return fig
    
    @log_function_call
    def generate_interpretation_report(self, model: Any, X_test: pd.DataFrame,
                                     y_test: pd.Series, task_type: str,
                                     model_name: str) -> Dict[str, Any]:
        """Generate comprehensive interpretation report with enhanced analysis"""
        report = {
            'model_name': model_name,
            'interpretability_methods': [],
            'feature_importance': {},
            'insights': [],
            'debug_info': {}
        }
        
        # Enhanced SHAP analysis
        if self.config.features.shap_analysis and self.shap_available:
            logger.info("Calculating SHAP values with enhanced analysis...")
            shap_values = self.calculate_shap_values(model, X_test, task_type)
            
            if shap_values is not None:
                # Validate SHAP values
                is_valid, diagnostics = self.validate_shap_values(shap_values, model_name)
                
                report['shap_values'] = shap_values
                report['shap_diagnostics'] = diagnostics
                report['shap_validated'] = is_valid
                report['interpretability_methods'].append('SHAP')
                
                if is_valid:
                    # Calculate feature importance from SHAP
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    shap_importance = dict(zip(X_test.columns, mean_abs_shap))
                    report['feature_importance']['shap'] = shap_importance
                    
                    # Enhanced insights
                    top_features = sorted(shap_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:5]
                    report['insights'].append(
                        f"Top 5 SHAP features: {', '.join([f[0] for f in top_features])}"
                    )
                    
                    # Value range insights
                    range_analysis = self.range_analyzer.analyze_range(shap_values.flatten())
                    if range_analysis.get('max_abs', 0) < 1e-3:
                        report['insights'].append(
                            "SHAP values are very small - model has subtle feature effects"
                        )
                    elif range_analysis.get('max_abs', 0) > 1:
                        report['insights'].append(
                            "SHAP values are large - model has strong feature effects"
                        )
                else:
                    report['insights'].append(
                        f"SHAP validation failed: {', '.join(diagnostics.get('validation_reasons', []))}"
                    )
                    for suggestion in diagnostics.get('improvement_suggestions', []):
                        report['insights'].append(f"Suggestion: {suggestion}")
        
        # Permutation importance (always calculate)
        logger.info("Calculating permutation importance...")
        perm_importance = self.calculate_permutation_importance(model, X_test, y_test)
        
        if perm_importance:
            report['interpretability_methods'].append('Permutation Importance')
            report['feature_importance']['permutation'] = {
                k: v['importance_mean'] for k, v in perm_importance.items()
            }
            
            # Enhanced permutation insights
            negative_features = [k for k, v in perm_importance.items() 
                               if v['importance_mean'] < -0.001]
            if negative_features:
                report['insights'].append(
                    f"Features with negative permutation importance: {', '.join(negative_features[:3])}"
                )
            
            # Check importance magnitude
            max_importance = max((v['importance_mean'] for v in perm_importance.values()), default=0)
            if max_importance < 0.01:
                report['insights'].append(
                    "Low permutation importance values - features may have weak individual effects"
                )
        
        # Model-specific interpretability
        model_type = type(model).__name__
        
        if hasattr(model, 'feature_importances_'):
            report['interpretability_methods'].append('Built-in Feature Importance')
            report['feature_importance']['built_in'] = dict(
                zip(X_test.columns, model.feature_importances_)
            )
            
            # Tree-based insights
            zero_importance = np.sum(model.feature_importances_ == 0)
            if zero_importance > 0:
                report['insights'].append(
                    f"Tree model assigned zero importance to {zero_importance} features"
                )
        
        elif hasattr(model, 'coef_'):
            report['interpretability_methods'].append('Linear Coefficients')
            if len(model.coef_.shape) == 1:
                report['feature_importance']['coefficients'] = dict(
                    zip(X_test.columns, np.abs(model.coef_))
                )
            else:
                # Multi-class - average absolute coefficients
                report['feature_importance']['coefficients'] = dict(
                    zip(X_test.columns, np.abs(model.coef_).mean(axis=0))
                )
        
        # Cross-method comparison
        if len(report['feature_importance']) > 1:
            report['insights'].append("Multiple importance methods available for comparison")
            
            # Find features consistently important across methods
            all_methods = report['feature_importance']
            consistent_features = []
            
            if len(all_methods) >= 2:
                # Get top 5 features from each method
                method_tops = {}
                for method_name, importances in all_methods.items():
                    top_5 = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    method_tops[method_name] = set(f[0] for f in top_5)
                
                # Find intersection
                if len(method_tops) >= 2:
                    consistent = set.intersection(*method_tops.values())
                    if consistent:
                        report['insights'].append(
                            f"Consistently important features: {', '.join(list(consistent))}"
                        )
        
        logger.info(f"Interpretation report generated with {len(report['interpretability_methods'])} methods")
        
        return report
    
    def create_interpretation_dashboard(self, interpretation_report: Dict[str, Any],
                                      X_test: pd.DataFrame) -> go.Figure:
        """Create comprehensive interpretation dashboard with enhanced formatting"""
        # Create subplots with enhanced layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance Comparison', 
                          'Method Agreement Analysis',
                          'Importance Distribution',
                          'Model Diagnostics'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "table"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        feature_importance = interpretation_report.get('feature_importance', {})
        
        if feature_importance:
            # 1. Feature importance comparison
            methods = list(feature_importance.keys())
            all_features = set()
            for importances in feature_importance.values():
                all_features.update(importances.keys())
            
            # Get top features overall
            feature_scores = {}
            for feature in all_features:
                scores = [abs(feature_importance[method].get(feature, 0)) for method in methods]
                feature_scores[feature] = np.mean(scores)
            
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:15]
            
            for i, method in enumerate(methods):
                importances = [abs(feature_importance[method].get(f[0], 0)) for f in top_features]
                
                fig.add_trace(go.Bar(
                    x=[f[0] for f in top_features],
                    y=importances,
                    name=method.capitalize(),
                    offsetgroup=i
                ), row=1, col=1)
        
        # Enhanced layout
        fig.update_layout(
            title_text=f"Enhanced Model Interpretation Dashboard - {interpretation_report['model_name']}",
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
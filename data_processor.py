"""
Data Processing Module for AquaVista v7.0 - DEBUG VERSION
========================================
Handles all data preprocessing, cleaning, and feature engineering operations.
Enhanced with comprehensive debugging to identify sign inversion issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                 LabelEncoder, OneHotEncoder)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression,
                                     mutual_info_classif, mutual_info_regression,
                                     RFE, SelectFromModel)
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
import logging
from pathlib import Path
from datetime import datetime
import joblib
import re

# Import custom modules
from modules.config import Config, DataProcessingError
from modules.logging_config import get_logger

# Safe import of core improvements with multicollinearity support
try:
    from modules.core_improvements import (
        MemoryOptimizer, SmartCache, UnifiedPipeline, 
        handle_class_imbalance, TargetEncoder, FeatureNameSanitizer,
        MulticollinearityHandler
    )
    MULTICOLLINEARITY_AVAILABLE = True
except ImportError:
    try:
        from modules.core_improvements import (
            MemoryOptimizer, SmartCache, UnifiedPipeline, 
            handle_class_imbalance, TargetEncoder, FeatureNameSanitizer
        )
        MULTICOLLINEARITY_AVAILABLE = False
        print("[WARNING] MulticollinearityHandler not available - statsmodels may not be installed")
    except ImportError as e:
        print(f"[ERROR] Core improvements module not available: {e}")
        # Create minimal fallbacks
        class MemoryOptimizer:
            @staticmethod
            def optimize_dtypes(df, verbose=True):
                return df
        
        class SmartCache:
            def __init__(self):
                self.cache_stats = {"hits": 0, "misses": 0}
            def cache_result(self, expire_hours=24):
                def decorator(func):
                    return func
                return decorator
        
        class UnifiedPipeline:
            def __init__(self):
                pass
        
        def handle_class_imbalance(X, y, strategy='auto'):
            return X, y, None
        
        class TargetEncoder:
            def __init__(self):
                self.encoder = None
                self.classes_ = None
            def fit_transform(self, y):
                return y
            def inverse_transform(self, y):
                return y
        
        class FeatureNameSanitizer:
            def __init__(self):
                pass
            def needs_sanitization(self, names):
                return False
            def sanitize_dataframe(self, df):
                return df
        
        MULTICOLLINEARITY_AVAILABLE = False

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

# Initialize cache
cache = SmartCache()

class PipelineDebugger:
    """Comprehensive debugging utilities for the data processing pipeline"""
    
    def __init__(self, enable_debug: bool = True):
        self.enable_debug = enable_debug
        self.correlation_history = []
        self.step_counter = 0
        
    def log_correlations(self, X: pd.DataFrame, y: pd.Series, step_name: str, 
                        target_column: str = "target"):
        """Track correlations at each pipeline step"""
        if not self.enable_debug:
            return
            
        try:
            # Calculate correlations
            correlations = X.corrwith(y)
            
            # Get top positive and negative correlations
            pos_corr = correlations.nlargest(5)
            neg_corr = correlations.nsmallest(5)
            
            logger.info(f"=== CORRELATION TRACKING: {step_name} ===")
            logger.info(f"Step {self.step_counter}: {step_name}")
            logger.info(f"Data shape: {X.shape}")
            
            logger.info("Top 5 POSITIVE correlations:")
            for feat, corr in pos_corr.items():
                logger.info(f"  {feat}: +{corr:.6f}")
            
            logger.info("Top 5 NEGATIVE correlations:")
            for feat, corr in neg_corr.items():
                logger.info(f"  {feat}: {corr:.6f}")
            
            # Store for comparison
            correlation_snapshot = {
                'step': self.step_counter,
                'step_name': step_name,
                'correlations': correlations.to_dict(),
                'data_shape': X.shape
            }
            self.correlation_history.append(correlation_snapshot)
            
            # Compare with previous step if available
            if len(self.correlation_history) > 1:
                self._compare_correlations(step_name)
            
            logger.info(f"=== END CORRELATION TRACKING ===\n")
            self.step_counter += 1
            
        except Exception as e:
            logger.error(f"Error in correlation tracking: {e}")
    
    def _compare_correlations(self, current_step: str):
        """Compare correlations with previous step"""
        if len(self.correlation_history) < 2:
            return
            
        current = self.correlation_history[-1]['correlations']
        previous = self.correlation_history[-2]['correlations']
        
        # Find features that exist in both steps
        common_features = set(current.keys()) & set(previous.keys())
        
        sign_flips = []
        large_changes = []
        
        for feature in common_features:
            curr_corr = current[feature]
            prev_corr = previous[feature]
            
            # Check for sign flips
            if (curr_corr > 0) != (prev_corr > 0) and abs(curr_corr) > 0.01 and abs(prev_corr) > 0.01:
                sign_flips.append({
                    'feature': feature,
                    'previous': prev_corr,
                    'current': curr_corr,
                    'change': curr_corr - prev_corr
                })
            
            # Check for large changes (>50% change)
            if abs(prev_corr) > 0.01:  # Only check meaningful correlations
                change_pct = abs((curr_corr - prev_corr) / prev_corr)
                if change_pct > 0.5:
                    large_changes.append({
                        'feature': feature,
                        'previous': prev_corr,
                        'current': curr_corr,
                        'change_pct': change_pct * 100
                    })
        
        # Report issues
        if sign_flips:
            logger.error(f"🚨 SIGN FLIPS DETECTED in step '{current_step}':")
            for flip in sign_flips[:5]:  # Show first 5
                logger.error(f"  {flip['feature']}: {flip['previous']:.6f} -> {flip['current']:.6f}")
        
        if large_changes:
            logger.warning(f"⚠️  LARGE CORRELATION CHANGES in step '{current_step}':")
            for change in large_changes[:5]:  # Show first 5
                logger.warning(f"  {change['feature']}: {change['previous']:.6f} -> {change['current']:.6f} ({change['change_pct']:.1f}% change)")
    
    def verify_index_alignment(self, X: pd.DataFrame, y: pd.Series, step_name: str):
        """Verify that X and y indices are aligned"""
        if not self.enable_debug:
            return
            
        logger.info(f"=== INDEX VERIFICATION: {step_name} ===")
        
        # Check basic alignment
        if len(X) != len(y):
            logger.error(f"❌ LENGTH MISMATCH: X has {len(X)} rows, y has {len(y)} rows")
            return False
        
        # Check index alignment
        if not X.index.equals(y.index):
            logger.error(f"❌ INDEX MISALIGNMENT detected!")
            logger.error(f"X index range: {X.index.min()} to {X.index.max()}")
            logger.error(f"y index range: {y.index.min()} to {y.index.max()}")
            
            # Show first few mismatched indices
            x_indices = set(X.index)
            y_indices = set(y.index)
            x_only = x_indices - y_indices
            y_only = y_indices - x_indices
            
            if x_only:
                logger.error(f"X-only indices (first 10): {list(x_only)[:10]}")
            if y_only:
                logger.error(f"y-only indices (first 10): {list(y_only)[:10]}")
            
            return False
        
        logger.info(f"✅ Index alignment verified: {len(X)} rows perfectly aligned")
        return True
    
    def log_feature_engineering_impact(self, X_before: pd.DataFrame, X_after: pd.DataFrame, 
                                     y: pd.Series, step_name: str):
        """Log the impact of feature engineering"""
        if not self.enable_debug:
            return
            
        logger.info(f"=== FEATURE ENGINEERING IMPACT: {step_name} ===")
        
        # Basic stats
        features_before = set(X_before.columns)
        features_after = set(X_after.columns)
        new_features = features_after - features_before
        removed_features = features_before - features_after
        
        logger.info(f"Features before: {len(features_before)}")
        logger.info(f"Features after: {len(features_after)}")
        logger.info(f"New features: {len(new_features)}")
        logger.info(f"Removed features: {len(removed_features)}")
        
        if new_features:
            logger.info(f"New feature names: {list(new_features)[:10]}...")
        
        if removed_features:
            logger.warning(f"Removed features: {list(removed_features)}")
        
        # Check correlations of original features
        original_features_remaining = features_before & features_after
        if original_features_remaining:
            logger.info("Correlation changes for original features:")
            
            for feature in list(original_features_remaining)[:5]:
                corr_before = X_before[feature].corr(y)
                corr_after = X_after[feature].corr(y)
                
                if abs(corr_before) > 0.01:  # Only check meaningful correlations
                    change = corr_after - corr_before
                    status = "🚨 SIGN FLIP" if (corr_before > 0) != (corr_after > 0) else "✅ OK"
                    logger.info(f"  {feature}: {corr_before:.6f} -> {corr_after:.6f} ({change:+.6f}) {status}")
        
        logger.info(f"=== END FEATURE ENGINEERING IMPACT ===\n")
    
    def validate_feature_names(self, X: pd.DataFrame, step_name: str):
        """Validate feature names for potential issues"""
        if not self.enable_debug:
            return
            
        logger.info(f"=== FEATURE NAME VALIDATION: {step_name} ===")
        
        issues = []
        
        # Check for duplicate column names
        duplicates = X.columns[X.columns.duplicated()].tolist()
        if duplicates:
            issues.append(f"Duplicate columns: {duplicates}")
        
        # Check for empty/null column names
        empty_names = [i for i, col in enumerate(X.columns) if not col or str(col).strip() == '']
        if empty_names:
            issues.append(f"Empty column names at positions: {empty_names}")
        
        # Check for problematic characters
        problematic = [col for col in X.columns if not str(col).replace('_', '').replace('.', '').isalnum()]
        if problematic:
            issues.append(f"Problematic column names: {problematic[:5]}...")
        
        if issues:
            logger.warning("Feature name issues found:")
            for issue in issues:
                logger.warning(f"  ⚠️  {issue}")
        else:
            logger.info("✅ All feature names are valid")
        
        logger.info(f"Total features: {len(X.columns)}")
        logger.info(f"=== END FEATURE NAME VALIDATION ===\n")


class DataProcessor:
    """Handles all data preprocessing operations with comprehensive debugging"""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessors = {}
        self.encoders = {}
        self.feature_names = []
        self.processing_report = {}
        self.pipeline = None
        self.target_encoder = None
        self.feature_sanitizer = None
        self.debugger = PipelineDebugger(enable_debug=True)  # Enable debugging
        
        # Initialize multicollinearity handler
        if MULTICOLLINEARITY_AVAILABLE:
            self.multicollinearity_handler = MulticollinearityHandler(
                vif_threshold=config.multicollinearity.get('vif_threshold', 10.0),
                correlation_threshold=config.multicollinearity.get('correlation_threshold', 0.9)
            )
            logger.info("Multicollinearity handler initialized")
        else:
            self.multicollinearity_handler = None
            logger.warning("Multicollinearity handler not available - statsmodels may not be installed")
        
    def process_data(self, df: pd.DataFrame, target_column: str,
                    feature_columns: List[str] = None,
                    preprocess_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main data processing pipeline with comprehensive debugging"""
        
        logger.info(f"Starting data processing for target: {target_column}")
        logger.info(f"=== PIPELINE DEBUGGING ENABLED ===")
        
        # Step 1: Optimize memory usage if enabled
        if getattr(self.config.features, 'memory_optimization', True):
            logger.info("Optimizing memory usage...")
            df = MemoryOptimizer.optimize_dtypes(df, verbose=True)
        
        # Validate inputs
        if target_column not in df.columns:
            raise DataProcessingError(f"Target column '{target_column}' not found in data")
        
        # Use all columns except target if features not specified
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Default preprocessing config
        if preprocess_config is None:
            preprocess_config = self._get_default_preprocess_config()
        
        # Initialize components
        self.pipeline = UnifiedPipeline()
        self.target_encoder = TargetEncoder()
        self.feature_sanitizer = FeatureNameSanitizer()
        
        # Initialize report
        self.processing_report = {
            'original_shape': df.shape,
            'steps': [],
            'warnings': [],
            'memory_saved': 0
        }
        
        try:
            # 1. Initial data validation
            self._validate_data(df, target_column, feature_columns)
            
            # 2. Separate features and target
            feature_data = df[feature_columns].copy()
            target_data = df[target_column].copy()
            
            # DEBUG CHECKPOINT 1: Raw data correlations
            logger.info("🔍 DEBUG CHECKPOINT 1: Raw Data")
            self.debugger.log_correlations(feature_data, target_data, "1_RAW_DATA")
            self.debugger.verify_index_alignment(feature_data, target_data, "1_RAW_DATA")
            
            # 3. Determine task type
            task_type = self._determine_task_type(target_data)
            self.processing_report['task_type'] = task_type
            logger.info(f"Detected task type: {task_type}")
            
            # 4. Handle missing values in target
            if target_data.isnull().any():
                logger.warning(f"Found {target_data.isnull().sum()} missing values in target")
                valid_indices = target_data.notna()
                feature_data = feature_data[valid_indices]
                target_data = target_data[valid_indices]
                self.processing_report['steps'].append({
                    'step': 'Remove missing targets',
                    'removed_rows': (~valid_indices).sum()
                })
                
                # DEBUG CHECKPOINT 2: After removing missing targets
                logger.info("🔍 DEBUG CHECKPOINT 2: After Target Cleaning")
                self.debugger.log_correlations(feature_data, target_data, "2_TARGET_CLEANED")
                self.debugger.verify_index_alignment(feature_data, target_data, "2_TARGET_CLEANED")
            
            # 5. Encode categorical target if needed
            if task_type == 'classification' and target_data.dtype in ['object', 'category']:
                logger.info("Encoding categorical target variable")
                
                # DEBUG: Check target before encoding
                logger.info(f"Target before encoding: {target_data.value_counts().to_dict()}")
                
                target_data_encoded = self.target_encoder.fit_transform(target_data)
                
                # DEBUG: Check target after encoding
                logger.info(f"Target after encoding: {pd.Series(target_data_encoded).value_counts().to_dict()}")
                logger.info(f"Target encoder classes: {self.target_encoder.classes_}")
                
                # CRITICAL CHECK: Ensure encoding preserves relationships
                if len(np.unique(target_data_encoded)) != len(self.target_encoder.classes_):
                    logger.error("❌ Target encoding error: class count mismatch!")
                
                target_data = pd.Series(target_data_encoded, index=target_data.index)
                self.encoders['target'] = self.target_encoder
                
                self.processing_report['steps'].append({
                    'step': 'Encode target',
                    'classes': list(self.target_encoder.classes_),
                    'n_classes': len(self.target_encoder.classes_)
                })
                
                # DEBUG CHECKPOINT 3: After target encoding
                logger.info("🔍 DEBUG CHECKPOINT 3: After Target Encoding")
                self.debugger.log_correlations(feature_data, target_data, "3_TARGET_ENCODED")
                self.debugger.verify_index_alignment(feature_data, target_data, "3_TARGET_ENCODED")
            
            # 6. Identify feature types
            numerical_features = feature_data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            categorical_features = feature_data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            logger.info(f"Found {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
            
            # 7. Create preprocessing pipeline
            self.pipeline.create_preprocessor(
                feature_data,
                numeric_features=numerical_features,
                categorical_features=categorical_features
            )
            
            # 8. Handle categorical features (if manual encoding requested)
            if categorical_features and preprocess_config.get('encoding_method') != 'auto':
                feature_data_before_encoding = feature_data.copy()
                
                feature_data = self._handle_categorical_features(
                    feature_data, categorical_features, 
                    preprocess_config.get('encoding_method', 'auto')
                )
                
                # DEBUG CHECKPOINT 4: After categorical encoding
                logger.info("🔍 DEBUG CHECKPOINT 4: After Categorical Encoding")
                self.debugger.log_feature_engineering_impact(
                    feature_data_before_encoding, feature_data, target_data, 
                    "4_CATEGORICAL_ENCODED"
                )
                self.debugger.log_correlations(feature_data, target_data, "4_CATEGORICAL_ENCODED")
                self.debugger.verify_index_alignment(feature_data, target_data, "4_CATEGORICAL_ENCODED")
                
                # Update feature lists after encoding
                numerical_features = feature_data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
                categorical_features = feature_data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 9. Handle missing values
            feature_data_before_missing = feature_data.copy()
            
            feature_data = self._handle_missing_values(
                feature_data, numerical_features, 
                preprocess_config.get('missing_strategy', 'auto')
            )
            
            # DEBUG CHECKPOINT 5: After handling missing values
            if not feature_data_before_missing.equals(feature_data):
                logger.info("🔍 DEBUG CHECKPOINT 5: After Missing Value Handling")
                self.debugger.log_correlations(feature_data, target_data, "5_MISSING_HANDLED")
                self.debugger.verify_index_alignment(feature_data, target_data, "5_MISSING_HANDLED")
            
            # 10. Remove outliers if requested
            if preprocess_config.get('remove_outliers', False):
                feature_data, target_data = self._remove_outliers(
                    feature_data, target_data, numerical_features,
                    preprocess_config.get('outlier_threshold', 1.5)
                )
                
                # DEBUG CHECKPOINT 6: After outlier removal
                logger.info("🔍 DEBUG CHECKPOINT 6: After Outlier Removal")
                self.debugger.log_correlations(feature_data, target_data, "6_OUTLIERS_REMOVED")
                self.debugger.verify_index_alignment(feature_data, target_data, "6_OUTLIERS_REMOVED")
            
            # ==================================================================
            # STEP 10A: Multicollinearity Diagnostics (NO AUTOMATIC REMOVAL)
            # ==================================================================
            removed_early = []  # Keep all features for now

            # Calculate diagnostics for user guidance
            mc_diagnostics = {
                'correlation_matrix': None,
                'high_corr_pairs': [],
                'vif_scores': {},
                'severity': 'unknown',
                'recommended_correlation_threshold': 0.90,
                'recommended_vif_threshold': 10.0,
                'guidance': ''
            }

            try:
                num_cols = feature_data.select_dtypes(include=['number']).columns.tolist()
                
                if len(num_cols) > 1:
                    # Correlation analysis
                    corr_matrix = feature_data[num_cols].corr()
                    mc_diagnostics['correlation_matrix'] = corr_matrix
                    
                    # Find high correlation pairs
                    for i in range(len(num_cols)):
                        for j in range(i + 1, len(num_cols)):
                            corr_val = abs(corr_matrix.iloc[i, j])
                            if not np.isnan(corr_val) and corr_val > 0.5:  # Only show meaningful correlations
                                mc_diagnostics['high_corr_pairs'].append({
                                    'feature_1': num_cols[i],
                                    'feature_2': num_cols[j],
                                    'correlation': float(corr_val)
                                })
                    
                    # Sort by correlation strength
                    mc_diagnostics['high_corr_pairs'].sort(key=lambda x: x['correlation'], reverse=True)
                    
                    # VIF calculation (if enough features)
                    if len(num_cols) > 2:
                        from statsmodels.stats.outliers_influence import variance_inflation_factor
                        
                        X_vif = feature_data[num_cols].dropna()
                        if len(X_vif) > len(num_cols):  # Need more samples than features
                            try:
                                vif_data = pd.DataFrame()
                                vif_data["Feature"] = num_cols
                                vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) 
                                                for i in range(len(num_cols))]
                                
                                mc_diagnostics['vif_scores'] = dict(zip(
                                    vif_data["Feature"], 
                                    vif_data["VIF"].values
                                ))
                            except Exception as e:
                                logger.warning(f"VIF calculation failed: {e}")
                    
                    # Assess severity and provide guidance
                    max_corr = max([p['correlation'] for p in mc_diagnostics['high_corr_pairs']], default=0)
                    max_vif = max(mc_diagnostics['vif_scores'].values(), default=0) if mc_diagnostics['vif_scores'] else 0
                    
                    if max_vif > 20 or max_corr > 0.95:
                        mc_diagnostics['severity'] = 'high'
                        mc_diagnostics['recommended_correlation_threshold'] = 0.90
                        mc_diagnostics['recommended_vif_threshold'] = 10.0
                        mc_diagnostics['guidance'] = (
                            f"HIGH multicollinearity detected (max |r|={max_corr:.2f}, max VIF={max_vif:.1f}). "
                            "Recommended: Enable multicollinearity handling with correlation threshold 0.90 and VIF threshold 10."
                        )
                    elif max_vif > 10 or max_corr > 0.85:
                        mc_diagnostics['severity'] = 'moderate'
                        mc_diagnostics['recommended_correlation_threshold'] = 0.92
                        mc_diagnostics['recommended_vif_threshold'] = 15.0
                        mc_diagnostics['guidance'] = (
                            f"MODERATE multicollinearity detected (max |r|={max_corr:.2f}, max VIF={max_vif:.1f}). "
                            "Consider enabling multicollinearity handling with correlation threshold 0.92 and VIF threshold 15."
                        )
                    else:
                        mc_diagnostics['severity'] = 'low'
                        mc_diagnostics['recommended_correlation_threshold'] = 0.95
                        mc_diagnostics['recommended_vif_threshold'] = 20.0
                        mc_diagnostics['guidance'] = (
                            f"LOW multicollinearity (max |r|={max_corr:.2f}, max VIF={max_vif:.1f}). "
                            "Multicollinearity handling not required, but you can enable it with threshold 0.95 for safety."
                        )
                    
                    logger.info(f"Multicollinearity diagnostics: {mc_diagnostics['severity']} severity")
                    
            except Exception as e:
                logger.warning(f"Multicollinearity diagnostics failed: {e}")

            self.processing_report['multicollinearity_diagnostics'] = mc_diagnostics

                
            # 11. Feature engineering with caching (ENHANCED: correlation-aware)
            if preprocess_config.get('feature_engineering', True):
                original_features = len(feature_data.columns)
                feature_data_before_engineering = feature_data.copy()
                
                # Check existing correlation levels to limit feature engineering if needed
                max_corr = feature_data.corr().abs().values[np.triu_indices_from(feature_data.corr().abs().values, k=1)].max() if len(feature_data.columns) > 1 else 0
                
                # Use caching for expensive feature engineering
                @cache.cache_result(expire_hours=24)
                def cached_feature_engineering(df_hash, columns, feature_limit):
                    return self.create_polynomial_features(
                        feature_data, 
                        columns[:feature_limit]
                    )
                
                # Adjust feature engineering based on correlation
                if max_corr > 0.85:
                    logger.info(f"High existing correlation ({max_corr:.3f}). Using conservative feature engineering.")
                    feature_limit = min(3, len(numerical_features))
                else:
                    feature_limit = min(10, len(numerical_features))
                
                # Create hash of dataframe for caching
                df_hash = pd.util.hash_pandas_object(feature_data).sum()
                feature_data = cached_feature_engineering(df_hash, numerical_features, feature_limit)
                
                engineered_features = len(feature_data.columns)
                
                logger.info(f"Feature engineering: {original_features} -> {engineered_features} features")
                
                # CRITICAL DEBUG CHECKPOINT 7: After feature engineering
                logger.info("🔍 DEBUG CHECKPOINT 7: After Feature Engineering (CRITICAL)")
                self.debugger.log_feature_engineering_impact(
                    feature_data_before_engineering, feature_data, target_data,
                    "7_FEATURE_ENGINEERED"
                )
                self.debugger.log_correlations(feature_data, target_data, "7_FEATURE_ENGINEERED")
                self.debugger.verify_index_alignment(feature_data, target_data, "7_FEATURE_ENGINEERED")
                self.debugger.validate_feature_names(feature_data, "7_FEATURE_ENGINEERED")
                
                self.processing_report['steps'].append({
                    'step': 'Feature engineering',
                    'original_features': original_features,
                    'engineered_features': engineered_features,
                    'new_features': engineered_features - original_features
                })
                
                # Update feature lists
                numerical_features = feature_data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            
            # 12. Feature selection if too many features
            if len(feature_data.columns) > 50:
                feature_data_before_selection = feature_data.copy()
                
                feature_data = self._feature_selection(
                    feature_data, target_data, task_type,
                    max_features=50
                )
                
                # DEBUG CHECKPOINT 8: After feature selection
                logger.info("🔍 DEBUG CHECKPOINT 8: After Feature Selection")
                self.debugger.log_feature_engineering_impact(
                    feature_data_before_selection, feature_data, target_data,
                    "8_FEATURE_SELECTED"
                )
                self.debugger.log_correlations(feature_data, target_data, "8_FEATURE_SELECTED")
                self.debugger.verify_index_alignment(feature_data, target_data, "8_FEATURE_SELECTED")
            
            # 13. Handle class imbalance for classification
            class_weights = None
            if task_type == 'classification' and preprocess_config.get('handle_imbalance', True):
                logger.info("Checking for class imbalance...")
                feature_data, target_data, class_weights = handle_class_imbalance(
                    feature_data, target_data, strategy='auto'
                )
                if class_weights:
                    self.processing_report['steps'].append({
                        'step': 'Handle class imbalance',
                        'method': 'class_weights',
                        'weights': class_weights
                    })
                    
                    # DEBUG CHECKPOINT 9: After class imbalance handling
                    logger.info("🔍 DEBUG CHECKPOINT 9: After Class Imbalance Handling")
                    self.debugger.log_correlations(feature_data, target_data, "9_IMBALANCE_HANDLED")
                    self.debugger.verify_index_alignment(feature_data, target_data, "9_IMBALANCE_HANDLED")
            
            # 14. Split data
            test_size = preprocess_config.get('test_size', 0.2)
            stratify = target_data if task_type == 'classification' and len(np.unique(target_data)) < 20 else None
            
            logger.info(f"Splitting data: test_size={test_size}, stratify={'Yes' if stratify is not None else 'No'}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                feature_data, target_data,
                test_size=test_size,
                random_state=self.config.computation.random_state,
                stratify=stratify
            )
            
            # DEBUG CHECKPOINT 10: After train/test split
            logger.info("🔍 DEBUG CHECKPOINT 10: After Train/Test Split")
            logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Verify split didn't break relationships
            X_combined_after_split = pd.concat([X_train, X_test])
            y_combined_after_split = pd.concat([y_train, y_test])
            
            self.debugger.log_correlations(X_combined_after_split, y_combined_after_split, "10_AFTER_SPLIT")
            self.debugger.verify_index_alignment(X_train, y_train, "10_TRAIN_SET")
            self.debugger.verify_index_alignment(X_test, y_test, "10_TEST_SET")
            
            # 15. Scale features (will be done per-model if auto)
            scaling_method = preprocess_config.get('scaling_method', 'auto')
            if scaling_method == 'auto (best for each model)':
                scaling_method = 'auto'
            
            if scaling_method == 'auto':
                logger.info('Skipping preprocessing scaling - will apply per-model scaling during training')
                self.processing_report['steps'].append({
                    'step': 'Scale features',
                    'method': 'auto (per-model)',
                    'status': 'deferred to model training'
                })
            elif scaling_method != 'none':
                X_train_before_scaling = X_train.copy()
                X_test_before_scaling = X_test.copy()
                
                X_train, X_test, scaler = self._scale_features(
                    X_train, X_test, numerical_features, scaling_method
                )
                self.preprocessors['scaler'] = scaler
                
                # DEBUG CHECKPOINT 11: After scaling
                logger.info("🔍 DEBUG CHECKPOINT 11: After Scaling")
                X_combined_scaled = pd.concat([X_train, X_test])
                y_combined = pd.concat([y_train, y_test])
                self.debugger.log_correlations(X_combined_scaled, y_combined, "11_SCALED")
            
            # 16. Store and sanitize feature names
            # Check if sanitization is needed
            if self.feature_sanitizer.needs_sanitization(X_train.columns.tolist()):
                logger.info("Sanitizing feature names for model compatibility...")
                
                X_train_before_sanitization = X_train.copy()
                X_test_before_sanitization = X_test.copy()
                
                # Sanitize the column names
                X_train = self.feature_sanitizer.sanitize_dataframe(X_train)
                X_test = self.feature_sanitizer.sanitize_dataframe(X_test)
                
                # Update numerical and categorical feature lists
                numerical_features = [self.feature_sanitizer.get_sanitized_name(f) for f in numerical_features if f in X_train.columns]
                categorical_features = [self.feature_sanitizer.get_sanitized_name(f) for f in categorical_features if f in X_train.columns]
                
                self.processing_report['steps'].append({
                    'step': 'Sanitize feature names',
                    'sanitized_count': len([name for name in self.feature_sanitizer.original_to_sanitized.values() if name != self.feature_sanitizer.get_original_name(name)])
                })
                
                # DEBUG CHECKPOINT 12: After sanitization
                logger.info("🔍 DEBUG CHECKPOINT 12: After Feature Name Sanitization")
                X_combined_sanitized = pd.concat([X_train, X_test])
                y_combined = pd.concat([y_train, y_test])
                self.debugger.log_correlations(X_combined_sanitized, y_combined, "12_SANITIZED")
                self.debugger.validate_feature_names(X_train, "12_SANITIZED")

            self.feature_names = list(X_train.columns)
            
            # 17. ENHANCED Multicollinearity Analysis and Treatment
            if preprocess_config.get('auto_handle_multicollinearity', False) and self.multicollinearity_handler:
                logger.info("Performing enhanced multicollinearity analysis...")
                
                # Update handler thresholds
                self.multicollinearity_handler.vif_threshold = preprocess_config.get('vif_threshold', 10.0)
                self.multicollinearity_handler.correlation_threshold = preprocess_config.get('correlation_threshold', 0.9)
                
                # Analyze multicollinearity on training set
                mc_analysis = self.multicollinearity_handler.analyze_multicollinearity(X_train)
                max_vif = max(mc_analysis.get("vif_scores", {}).values()) if mc_analysis.get("vif_scores") else 0
                
                logger.info(f"Multicollinearity analysis complete: severity = {mc_analysis.get('severity', 'unknown')}, max VIF = {max_vif:.2f}")
                
                # Apply enhanced treatment if needed (VIF > 10)
                if max_vif > 10:
                    logger.info(f"Elevated multicollinearity detected (VIF: {max_vif:.2f}). Applying enhanced treatment...")
                    
                    # Combine train and test for treatment
                    X_combined = pd.concat([X_train, X_test])
                    y_combined = pd.concat([y_train, y_test])
                    
                    # Apply treatment
                    X_treated, treatment_info = self.apply_multicollinearity_treatment_enhanced(
                        X_combined, y_combined, mc_analysis
                    )
                    
                    # Re-split the data
                    train_indices = X_train.index
                    test_indices = X_test.index
                    
                    X_train = X_treated.loc[train_indices]
                    X_test = X_treated.loc[test_indices]
                    
                    # Update feature lists
                    numerical_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
                    self.feature_names = list(X_train.columns)
                    
                    # Debug impact
                    logger.info(f"Features after treatment: {len(X_train.columns)} (removed {len(X_combined.columns) - len(X_train.columns)})")
                    
                    # Re-analyze after treatment
                    mc_analysis_after = self.multicollinearity_handler.analyze_multicollinearity(X_train)
                    max_vif_after = max(mc_analysis_after.get("vif_scores", {}).values()) if mc_analysis_after.get("vif_scores") else 0
                    
                    logger.info(f"Multicollinearity after treatment: severity = {mc_analysis_after.get('severity', 'unknown')}, max VIF = {max_vif_after:.2f}")
                    
                    # Store treatment results
                    mc_analysis['treatment_applied'] = treatment_info
                    mc_analysis['max_vif_before'] = max_vif
                    mc_analysis['max_vif_after'] = max_vif_after
                    mc_analysis['severity_after_treatment'] = mc_analysis_after.get('severity', 'unknown')
                
                # Store in processing report
                self.processing_report['multicollinearity_analysis'] = mc_analysis
                
                # DEBUG CHECKPOINT
                logger.info("🔍 DEBUG CHECKPOINT: After Enhanced Multicollinearity Treatment")

            # 18. Final validation
            self._validate_processed_data(X_train, X_test, y_train, y_test)
            
            # FINAL DEBUG CHECKPOINT: Complete pipeline
            logger.info("🔍 FINAL DEBUG CHECKPOINT: Complete Pipeline")
            X_final = pd.concat([X_train, X_test])
            y_final = pd.concat([y_train, y_test])
            self.debugger.log_correlations(X_final, y_final, "FINAL_PIPELINE")
            
            # Generate correlation comparison report
            self._generate_correlation_report()
            
            # Calculate memory savings
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            current_memory = (X_train.memory_usage(deep=True).sum() + 
                            X_test.memory_usage(deep=True).sum()) / 1024**2
            memory_saved = max(0, original_memory - current_memory)
            self.processing_report['memory_saved'] = memory_saved
            self.processing_report['config'] = preprocess_config
            
            # Create result dictionary
            result = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': self.feature_names,
                'feature_sanitizer': self.feature_sanitizer,
                'task_type': task_type,
                'n_classes': len(np.unique(target_data)) if task_type == 'classification' else None,
                'preprocessors': self.preprocessors,
                'encoders': self.encoders,
                'target_encoder': self.target_encoder,
                'class_weights': class_weights,
                'processing_report': self.processing_report,
                'numerical_features': [f for f in numerical_features if f in self.feature_names],
                'categorical_features': [f for f in categorical_features if f in self.feature_names],
                'scaling_method': scaling_method,
                'pipeline': self.pipeline,
                'target_column': target_column,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_features': len(self.feature_names),
                'correlation_history': self.debugger.correlation_history,  # Include debug history
                'multicollinearity_analysis': self.processing_report.get('multicollinearity_analysis', {})
            }
            
            logger.info(f"Processing complete. Shape: {X_train.shape}, Memory saved: {memory_saved:.1f}MB")
            logger.info("=== PIPELINE DEBUGGING COMPLETE ===")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            logger.error("Correlation history for debugging:")
            for entry in self.debugger.correlation_history:
                logger.error(f"  Step {entry['step']}: {entry['step_name']} - Shape: {entry['data_shape']}")
            raise DataProcessingError(f"Data processing failed: {str(e)}")
    
    def _generate_correlation_report(self):
        """Generate comprehensive correlation comparison report"""
        if len(self.debugger.correlation_history) < 2:
            return
            
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE CORRELATION CHANGE REPORT")
        logger.info("=" * 60)
        
        first_step = self.debugger.correlation_history[0]
        last_step = self.debugger.correlation_history[-1]
        
        first_corr = first_step['correlations']
        last_corr = last_step['correlations']
        
        # Find common features
        common_features = set(first_corr.keys()) & set(last_corr.keys())
        
        # Categorize changes
        sign_flips = []
        large_decreases = []
        large_increases = []
        stable_features = []
        
        for feature in common_features:
            first_val = first_corr[feature]
            last_val = last_corr[feature]
            
            # Skip very small correlations
            if abs(first_val) < 0.01 and abs(last_val) < 0.01:
                continue
            
            # Check for sign flip
            if (first_val > 0) != (last_val > 0) and abs(first_val) > 0.01 and abs(last_val) > 0.01:
                sign_flips.append({
                    'feature': feature,
                    'first': first_val,
                    'last': last_val,
                    'change': last_val - first_val
                })
            
            # Check for large changes
            elif abs(first_val) > 0.01:
                change_pct = (last_val - first_val) / first_val
                if change_pct < -0.5:
                    large_decreases.append({'feature': feature, 'first': first_val, 'last': last_val, 'change_pct': change_pct})
                elif change_pct > 0.5:
                    large_increases.append({'feature': feature, 'first': first_val, 'last': last_val, 'change_pct': change_pct})
                elif abs(change_pct) < 0.1:
                    stable_features.append({'feature': feature, 'first': first_val, 'last': last_val})
        
        # Report findings
        logger.info(f"ANALYSIS: {first_step['step_name']} -> {last_step['step_name']}")
        logger.info(f"Features compared: {len(common_features)}")
        
        if sign_flips:
            logger.error(f"🚨 CRITICAL: {len(sign_flips)} SIGN FLIPS DETECTED:")
            for flip in sorted(sign_flips, key=lambda x: abs(x['change']), reverse=True):
                logger.error(f"  {flip['feature']}: {flip['first']:+.6f} -> {flip['last']:+.6f} (Δ{flip['change']:+.6f})")
        
        if large_decreases:
            logger.warning(f"⚠️  {len(large_decreases)} features with large decreases:")
            for dec in sorted(large_decreases, key=lambda x: x['change_pct'])[:5]:
                logger.warning(f"  {dec['feature']}: {dec['first']:+.6f} -> {dec['last']:+.6f} ({dec['change_pct']*100:+.1f}%)")
        
        if large_increases:
            logger.warning(f"⬆️  {len(large_increases)} features with large increases:")
            for inc in sorted(large_increases, key=lambda x: x['change_pct'], reverse=True)[:5]:
                logger.warning(f"  {inc['feature']}: {inc['first']:+.6f} -> {inc['last']:+.6f} ({inc['change_pct']*100:+.1f}%)")
        
        if stable_features:
            logger.info(f"✅ {len(stable_features)} features remained stable (±10%)")
        
        logger.info("=" * 60)
    
    def _get_default_preprocess_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration"""
        return {
            'missing_strategy': 'auto',
            'scaling_method': 'auto',
            'encoding_method': 'auto',
            'remove_outliers': False,
            'outlier_threshold': 1.5,
            'test_size': 0.2,
            'feature_engineering': True,
            'handle_imbalance': True
        }
    
    def _validate_data(self, df: pd.DataFrame, target_column: str,
                      feature_columns: List[str]):
        """Validate input data"""
        # Check for empty dataframe
        if df.empty:
            raise DataProcessingError("Input dataframe is empty")
        
        # Check minimum samples
        if len(df) < 50:
            self.processing_report['warnings'].append(
                f"Small dataset detected ({len(df)} samples). Results may be unreliable."
            )
        
        # Check for missing columns
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise DataProcessingError(f"Feature columns not found: {missing_cols}")
        
        # Check for constant features
        constant_features = []
        for col in feature_columns:
            if df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            self.processing_report['warnings'].append(
                f"Constant features detected: {constant_features}. These will be removed."
            )
    
    def _determine_task_type(self, target: pd.Series) -> str:
        """Determine if task is classification or regression"""
        # Check data type
        if target.dtype == 'object' or target.dtype.name == 'category':
            return 'classification'
        
        # Check unique values
        n_unique = target.nunique()
        n_samples = len(target)
        
        # If very few unique values relative to samples, likely classification
        if n_unique < 20 or n_unique / n_samples < 0.05:
            return 'classification'
        
        # Check if values are integers and consecutive (likely classification)
        if target.dtype in ['int64', 'int32']:
            unique_vals = sorted(target.unique())
            if len(unique_vals) < 20:
                # Check if consecutive integers starting from 0 or 1
                if unique_vals == list(range(min(unique_vals), max(unique_vals) + 1)):
                    return 'classification'
        
        return 'regression'
    
    def _handle_categorical_features(self, df: pd.DataFrame, 
                                   categorical_columns: List[str],
                                   encoding_method: str) -> pd.DataFrame:
        """Handle categorical features with debugging"""
        df_encoded = df.copy()
        
        logger.info(f"Encoding {len(categorical_columns)} categorical features with method: {encoding_method}")
        
        for col in categorical_columns:
            n_unique = df[col].nunique()
            
            # Auto-select encoding method
            if encoding_method == 'auto':
                if n_unique <= 10:
                    method = 'onehot'
                else:
                    method = 'target'
            else:
                method = encoding_method
            
            logger.info(f"Encoding {col} ({n_unique} unique values) with {method}")
            
            if method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                
            elif method == 'ordinal':
                # Ordinal encoding
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col].fillna('missing'))
                self.encoders[col] = encoder
            
            elif method == 'target':
                # Simple frequency encoding for now
                freq_encoding = df[col].value_counts(normalize=True)
                df_encoded[col] = df[col].map(freq_encoding).fillna(0)
        
        self.processing_report['steps'].append({
            'step': 'Encode categorical',
            'method': encoding_method,
            'columns': categorical_columns,
            'final_columns': len(df_encoded.columns)
        })
        
        return df_encoded
    
    def _handle_missing_values(self, df: pd.DataFrame, 
                             numerical_columns: List[str],
                             strategy: str) -> pd.DataFrame:
        """Handle missing values with debugging"""
        df_filled = df.copy()
        
        # Check missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0].index.tolist()
        
        if not missing_cols:
            logger.info("No missing values found")
            return df_filled
        
        logger.info(f"Handling missing values in {len(missing_cols)} columns with strategy: {strategy}")
        
        # Auto-select strategy
        if strategy == 'auto':
            # Use median for numerical, mode for categorical
            strategy = 'median'
        
        # Apply strategy
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            # Only impute numerical columns
            num_cols_to_impute = [col for col in numerical_columns if col in df_filled.columns and col in missing_cols]
            if num_cols_to_impute:
                logger.info(f"Imputing {len(num_cols_to_impute)} numerical columns with {strategy}")
                df_filled[num_cols_to_impute] = imputer.fit_transform(df_filled[num_cols_to_impute])
                self.preprocessors['imputer'] = imputer
            
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            num_cols_to_impute = [col for col in numerical_columns if col in df_filled.columns]
            if num_cols_to_impute:
                logger.info(f"KNN imputing {len(num_cols_to_impute)} numerical columns")
                df_filled[num_cols_to_impute] = imputer.fit_transform(df_filled[num_cols_to_impute])
                self.preprocessors['imputer'] = imputer
            
        elif strategy == 'forward_fill':
            df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
            
        elif strategy == 'interpolate':
            df_filled[numerical_columns] = df_filled[numerical_columns].interpolate()
            
        elif strategy == 'drop':
            rows_before = len(df_filled)
            df_filled = df_filled.dropna()
            logger.info(f"Dropped {rows_before - len(df_filled)} rows with missing values")
        
        self.processing_report['steps'].append({
            'step': 'Handle missing values',
            'strategy': strategy,
            'missing_columns': missing_cols,
            'rows_before': len(df),
            'rows_after': len(df_filled)
        })
        
        return df_filled
    
    def _remove_outliers(self, X: pd.DataFrame, y: pd.Series,
                        numerical_columns: List[str],
                        threshold: float) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove outliers using IQR method with comprehensive debugging"""
        
        # DEBUGGING: Check alignment before outlier removal
        logger.info(f"=== OUTLIER REMOVAL DEBUG ===")
        logger.info(f"Before outlier removal - X shape: {X.shape}")
        logger.info(f"Before outlier removal - X index range: {X.index.min()}-{X.index.max()}")
        logger.info(f"Before outlier removal - y index range: {y.index.min()}-{y.index.max()}")
        logger.info(f"Before outlier removal - Index alignment: {X.index.equals(y.index)}")
        
        mask = pd.Series(True, index=X.index)
        outlier_details = {}
        
        for col in numerical_columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                col_outliers = ((X[col] < lower) | (X[col] > upper))
                col_mask = ~col_outliers
                mask = mask & col_mask
                
                outlier_count = col_outliers.sum()
                if outlier_count > 0:
                    outlier_details[col] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / len(X)) * 100,
                        'lower_bound': lower,
                        'upper_bound': upper
                    }
                    logger.info(f"  {col}: {outlier_count} outliers ({outlier_details[col]['percentage']:.1f}%)")
        
        outliers_removed = (~mask).sum()
        
        if outliers_removed > 0:
            logger.info(f"Total outliers to remove: {outliers_removed} ({outliers_removed/len(X)*100:.1f}%)")
            
            X_clean = X[mask]
            y_clean = y[mask]
            
            # DEBUGGING: Check alignment after outlier removal
            logger.info(f"After outlier removal - X shape: {X_clean.shape}")
            logger.info(f"After outlier removal - X index range: {X_clean.index.min()}-{X_clean.index.max()}")
            logger.info(f"After outlier removal - y index range: {y_clean.index.min()}-{y_clean.index.max()}")
            
            # Critical check: ensure indices still match
            if not X_clean.index.equals(y_clean.index):
                logger.error("❌ INDEX MISALIGNMENT DETECTED after outlier removal!")
                logger.error(f"X indices sample: {list(X_clean.index[:10])}")
                logger.error(f"y indices sample: {list(y_clean.index[:10])}")
                
                # Try to fix alignment
                common_indices = X_clean.index.intersection(y_clean.index)
                if len(common_indices) > 0:
                    logger.info(f"Attempting to fix alignment using {len(common_indices)} common indices")
                    X_clean = X_clean.loc[common_indices]
                    y_clean = y_clean.loc[common_indices]
                    logger.info("✅ Index alignment fixed")
                else:
                    raise DataProcessingError("Cannot fix index misalignment after outlier removal")
            else:
                logger.info("✅ Index alignment maintained after outlier removal")
            
            self.processing_report['steps'].append({
                'step': 'Remove outliers',
                'method': 'IQR',
                'threshold': threshold,
                'outliers_removed': outliers_removed,
                'percentage': f"{outliers_removed / len(X) * 100:.1f}%",
                'outlier_details': outlier_details
            })
            
            logger.info(f"=== END OUTLIER REMOVAL DEBUG ===")
            
            return X_clean, y_clean
        
        logger.info("No outliers detected")
        logger.info(f"=== END OUTLIER REMOVAL DEBUG ===")
        return X, y
    
        # ADD THE NEW METHOD HERE AT LINE 1068
    def _early_multicollinearity_check(self, X: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
        """Early detection and removal of obviously correlated features"""
        corr_matrix = X.corr().abs()
        
        features_to_remove = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    feat2 = corr_matrix.columns[j]
                    if feat2 not in features_to_remove:
                        features_to_remove.append(feat2)
        
        if features_to_remove:
            logger.info(f"Early removal of {len(features_to_remove)} highly correlated features")
            return X.drop(columns=features_to_remove), features_to_remove
        
        return X, []
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                columns: List[str]) -> pd.DataFrame:
        """Create polynomial and interaction features with comprehensive debugging"""
        logger.info(f"=== FEATURE ENGINEERING DEBUG ===")
        logger.info(f"Starting feature engineering with {len(df.columns)} features")
        logger.info(f"Numerical columns for engineering: {columns[:10]}...")
        
        df_poly = df.copy()
        original_columns = set(df.columns)
        
        def safe_feature_name(base_name: str, suffix: str = "") -> str:
            """Create safe feature names with comprehensive validation"""
            # Ensure base_name is valid
            if not base_name or str(base_name).strip() == '':
                base_name = f"Feature_{hash(str(base_name)) % 1000}"
            
            # Clean the name
            clean_name = str(base_name).strip()
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
            clean_name = re.sub(r'_+', '_', clean_name)
            clean_name = clean_name.strip('_')
            
            if suffix:
                clean_name = f"{clean_name}_{suffix}"
            
            # Ensure uniqueness
            counter = 0
            original_name = clean_name
            while clean_name in df_poly.columns:
                counter += 1
                clean_name = f"{original_name}_{counter}"
            
            return clean_name
        
        # Limit features to prevent explosion
        columns = columns[:min(10, len(columns))]
        logger.info(f"Limited to {len(columns)} columns for engineering")
        
        # Track new features
        interaction_features = []
        polynomial_features = []
        
        # Create interaction features (limited)
        logger.info("Creating interaction features...")
        for i in range(len(columns)):
            for j in range(i + 1, min(i + 5, len(columns))):  # Limit combinations
                col1, col2 = columns[i], columns[j]
                if col1 in df.columns and col2 in df.columns:
                    new_name = safe_feature_name(f"{col1}_x_{col2}")
                    
                    # Create interaction with error handling
                    try:
                        interaction_values = df[col1] * df[col2]
                        
                        # Validate interaction values
                        if not np.isfinite(interaction_values).all():
                            logger.warning(f"Non-finite values in interaction {new_name}, skipping")
                            continue
                            
                        df_poly[new_name] = interaction_values
                        interaction_features.append(new_name)
                        
                        logger.info(f"  Created interaction: {new_name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to create interaction {new_name}: {e}")
                        continue
        
        # Create polynomial features for top 5 columns only
        logger.info("Creating polynomial features...")
        for col in columns[:5]:
            if col in df.columns:
                try:
                    # Square
                    square_name = safe_feature_name(col, "sq")
                    square_values = df[col] ** 2
                    
                    if np.isfinite(square_values).all():
                        df_poly[square_name] = square_values
                        polynomial_features.append(square_name)
                        logger.info(f"  Created square: {square_name}")
                    else:
                        logger.warning(f"Non-finite values in square of {col}, skipping")
                    
                    # Square root (only for non-negative)
                    if (df[col] >= 0).all():
                        sqrt_name = safe_feature_name(col, "sqrt")
                        sqrt_values = np.sqrt(df[col])
                        
                        if np.isfinite(sqrt_values).all():
                            df_poly[sqrt_name] = sqrt_values
                            polynomial_features.append(sqrt_name)
                            logger.info(f"  Created sqrt: {sqrt_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to create polynomial features for {col}: {e}")
                    continue
        
        # Final validation
        new_columns = set(df_poly.columns) - original_columns
        logger.info(f"Feature engineering complete:")
        logger.info(f"  Original features: {len(original_columns)}")
        logger.info(f"  New features: {len(new_columns)}")
        logger.info(f"  Total features: {len(df_poly.columns)}")
        logger.info(f"  Interaction features: {len(interaction_features)}")
        logger.info(f"  Polynomial features: {len(polynomial_features)}")
        
        # Check for any problematic values in new features
        numeric_new_cols = df_poly[list(new_columns)].select_dtypes(include=[np.number]).columns
        if len(numeric_new_cols) > 0:
            inf_count = np.isinf(df_poly[numeric_new_cols]).sum().sum()
            nan_count = np.isnan(df_poly[numeric_new_cols]).sum().sum()
            
            if inf_count > 0:
                logger.error(f"❌ {inf_count} infinite values created in new features!")
            if nan_count > 0:
                logger.error(f"❌ {nan_count} NaN values created in new features!")
            
            if inf_count == 0 and nan_count == 0:
                logger.info("✅ All new features have finite values")
        
        logger.info(f"=== END FEATURE ENGINEERING DEBUG ===")
        
        return df_poly
    
    def _feature_selection(self, X: pd.DataFrame, y: pd.Series,
                         task_type: str, max_features: int) -> pd.DataFrame:
        """Select top features using mutual information with debugging"""
        logger.info(f"Performing feature selection: {len(X.columns)} -> {max_features} features")
        
        if task_type == 'classification':
            selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=max_features)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        removed_features = [col for col in X.columns if col not in selected_features]
        
        logger.info(f"Selected features: {selected_features[:10]}...")
        logger.info(f"Removed {len(removed_features)} features")
        
        self.processing_report['steps'].append({
            'step': 'Feature selection',
            'method': 'mutual_information',
            'features_before': len(X.columns),
            'features_after': len(selected_features),
            'selected_features': selected_features,
            'removed_features': removed_features[:20]  # Store first 20 removed features
        })
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
      # ADD THE NEW METHOD HERE AT LINE 1220
    def _ard_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                              relevance_threshold: float = 0.01) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Use ARD Regression for intelligent feature selection"""
        from sklearn.linear_model import ARDRegression
        
        logger.info("Applying ARD-based feature selection...")
        
        ard = ARDRegression(
            alpha_1=1e-6, alpha_2=1e-6, 
            lambda_1=1e-6, lambda_2=1e-6,
            threshold_lambda=1000,
            max_iter=300
        )
        
        ard.fit(X, y)
        feature_relevance = 1.0 / ard.lambda_
        relevant_mask = feature_relevance > relevance_threshold
        selected_features = X.columns[relevant_mask].tolist()

        # SAFETY CHECK: Never remove all features
        if len(selected_features) == 0:
            logger.warning("⚠️ ARD would remove all features! Keeping top 3 by absolute weight")
            feature_weights = np.abs(ard.coef_)
            top_3_idx = np.argsort(feature_weights)[-3:]
            selected_features = X.columns[top_3_idx].tolist()
            logger.info(f"Safety override: kept features {selected_features}")
        
        logger.info(f"ARD selected {len(selected_features)} out of {len(X.columns)} features")
        
        return X[selected_features], {
            'method': 'ARD',
            'original_features': len(X.columns),
            'selected_features': len(selected_features),
            'threshold': relevance_threshold,
            'ard_model': ard
        }

    # ADD THIS METHOD HERE AT LINE ~1271
    def apply_multicollinearity_treatment_enhanced(self, X: pd.DataFrame, y: pd.Series,
                                                  mc_analysis: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced treatment with automatic escalation"""
        severity = mc_analysis.get("severity", "low")
        max_vif = max(mc_analysis.get("vif_scores", {}).values()) if mc_analysis.get("vif_scores") else 0
        
        if max_vif > 50:  # Severe case
            logger.warning(f"Severe multicollinearity (VIF: {max_vif}). Applying ARD treatment.")
            return self._ard_feature_selection(X, y, relevance_threshold=0.01)
            
        elif max_vif > 20:  # High case
            X_ard, ard_info = self._ard_feature_selection(X, y, relevance_threshold=0.005)
            if len(X_ard.columns) > 15:
                # Additional correlation cleanup
                X_final, removed = self._early_multicollinearity_check(X_ard, threshold=0.95)
                ard_info['correlation_cleanup'] = removed
                return X_final, ard_info
            return X_ard, ard_info
            
        elif severity == "moderate":
            return self._early_multicollinearity_check(X, threshold=0.9)
            
        return X, {"treatment_applied": "none"}

    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       numerical_columns: List[str], 
                       method: str) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Scale numerical features with debugging"""
        # Get numerical columns that exist in the data
        num_cols_to_scale = [col for col in numerical_columns if col in X_train.columns]
        
        if not num_cols_to_scale:
            logger.info("No numerical columns found to scale")
            return X_train, X_test, None
        
        logger.info(f"Scaling {len(num_cols_to_scale)} numerical columns with method: {method}")
        
        # Auto-select scaler
        if method == 'auto':
            # Check for outliers
            outlier_ratio = 0
            for col in num_cols_to_scale[:10]:  # Check first 10 columns
                Q1 = X_train[col].quantile(0.25)
                Q3 = X_train[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = ((X_train[col] < Q1 - 1.5*IQR) | (X_train[col] > Q3 + 1.5*IQR)).sum()
                    outlier_ratio += outliers / len(X_train)
            
            outlier_ratio /= min(10, len(num_cols_to_scale))
            
            if outlier_ratio > 0.1:
                method = 'robust'
            else:
                method = 'standard'
            
            logger.info(f"Auto-selected scaler: {method} (outlier ratio: {outlier_ratio:.3f})")
        
        # Create scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return X_train, X_test, None
        
        # Fit and transform
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Log scaling statistics
        logger.info("Scaling statistics:")
        for col in num_cols_to_scale[:5]:  # Show first 5
            before_mean = X_train[col].mean()
            before_std = X_train[col].std()
            logger.info(f"  {col}: mean={before_mean:.3f}, std={before_std:.3f}")
        
        X_train_scaled[num_cols_to_scale] = scaler.fit_transform(X_train[num_cols_to_scale])
        X_test_scaled[num_cols_to_scale] = scaler.transform(X_test[num_cols_to_scale])
        
        # Log after scaling
        logger.info("After scaling:")
        for col in num_cols_to_scale[:5]:  # Show first 5
            after_mean = X_train_scaled[col].mean()
            after_std = X_train_scaled[col].std()
            logger.info(f"  {col}: mean={after_mean:.3f}, std={after_std:.3f}")
        
        self.processing_report['steps'].append({
            'step': 'Scale features',
            'method': method,
            'scaled_columns': num_cols_to_scale
        })
        
        return X_train_scaled, X_test_scaled, scaler
    
    def _validate_processed_data(self, X_train, X_test, y_train, y_test):
        """Validate processed data with comprehensive checks"""
        logger.info("=== FINAL DATA VALIDATION ===")
        
        # Check for NaN values
        train_nan_cols = X_train.isnull().any()
        test_nan_cols = X_test.isnull().any()
        
        if train_nan_cols.any():
            nan_columns = train_nan_cols[train_nan_cols].index.tolist()
            logger.error(f"❌ NaN values found in training features: {nan_columns}")
            raise DataProcessingError("NaN values found in training features after processing")
        
        if test_nan_cols.any():
            nan_columns = test_nan_cols[test_nan_cols].index.tolist()
            logger.error(f"❌ NaN values found in test features: {nan_columns}")
            raise DataProcessingError("NaN values found in test features after processing")
        
        # Check for infinite values only in numeric columns
        numeric_train_cols = X_train.select_dtypes(include=[np.number]).columns
        numeric_test_cols = X_test.select_dtypes(include=[np.number]).columns
        
        if len(numeric_train_cols) > 0:
            train_inf_count = np.isinf(X_train[numeric_train_cols].values).sum()
            if train_inf_count > 0:
                logger.error(f"❌ {train_inf_count} infinite values found in training features")
                raise DataProcessingError("Infinite values found in training features")
        
        if len(numeric_test_cols) > 0:
            test_inf_count = np.isinf(X_test[numeric_test_cols].values).sum()
            if test_inf_count > 0:
                logger.error(f"❌ {test_inf_count} infinite values found in test features")
                raise DataProcessingError("Infinite values found in test features")
        
        # Check shapes
        if len(X_train) != len(y_train):
            logger.error(f"❌ Shape mismatch: X_train({len(X_train)}) != y_train({len(y_train)})")
            raise DataProcessingError("Mismatch between X_train and y_train lengths")
        
        if len(X_test) != len(y_test):
            logger.error(f"❌ Shape mismatch: X_test({len(X_test)}) != y_test({len(y_test)})")
            raise DataProcessingError("Mismatch between X_test and y_test lengths")
        
        # Check minimum samples
        if len(X_train) < 10:
            logger.error(f"❌ Too few training samples: {len(X_train)}")
            raise DataProcessingError(f"Too few training samples: {len(X_train)}")
        
        # Check feature consistency
        if not X_train.columns.equals(X_test.columns):
            logger.error("❌ Feature columns don't match between train and test sets")
            train_only = set(X_train.columns) - set(X_test.columns)
            test_only = set(X_test.columns) - set(X_train.columns)
            if train_only:
                logger.error(f"Train-only columns: {train_only}")
            if test_only:
                logger.error(f"Test-only columns: {test_only}")
            raise DataProcessingError("Feature columns don't match between train and test sets")
        
        logger.info("✅ All validation checks passed")
        logger.info(f"Final shapes: X_train{X_train.shape}, X_test{X_test.shape}")
        logger.info(f"Feature count: {len(X_train.columns)}")
        logger.info("=== END FINAL DATA VALIDATION ===")
    
    def save_preprocessors(self, filepath: Path):
        """Save preprocessing objects"""
        save_data = {
            'preprocessors': self.preprocessors,
            'encoders': self.encoders,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names,
            'feature_sanitizer': self.feature_sanitizer,
            'processing_report': self.processing_report,
            'pipeline': self.pipeline,
            'correlation_history': self.debugger.correlation_history,
            'version': '7.0.0'
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Saved preprocessors to {filepath}")
    
    def load_preprocessors(self, filepath: Path) -> Dict[str, Any]:
        """Load preprocessing objects"""
        loaded_data = joblib.load(filepath)
        
        # Restore components
        self.preprocessors = loaded_data.get('preprocessors', {})
        self.encoders = loaded_data.get('encoders', {})
        self.target_encoder = loaded_data.get('target_encoder')
        self.feature_names = loaded_data.get('feature_names', [])
        self.feature_sanitizer = loaded_data.get('feature_sanitizer')
        self.processing_report = loaded_data.get('processing_report', {})
        self.pipeline = loaded_data.get('pipeline')
        
        # Restore debug history if available
        if 'correlation_history' in loaded_data:
            self.debugger.correlation_history = loaded_data['correlation_history']
        
        return loaded_data
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using saved preprocessors"""
        # Apply memory optimization
        if getattr(self.config.features, 'memory_optimization', True):
            df = MemoryOptimizer.optimize_dtypes(df, verbose=False)
        
        # Apply saved transformations
        df_transformed = df.copy()
        
        # Apply encoders
        for col, encoder in self.encoders.items():
            if col in df_transformed.columns:
                df_transformed[col] = encoder.transform(df_transformed[col])
        
        # Apply imputer
        if 'imputer' in self.preprocessors:
            imputer = self.preprocessors['imputer']
            numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
            df_transformed[numeric_cols] = imputer.transform(df_transformed[numeric_cols])
        
        # Apply scaler
        if 'scaler' in self.preprocessors:
            scaler = self.preprocessors['scaler']
            numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
            df_transformed[numeric_cols] = scaler.transform(df_transformed[numeric_cols])
        
        # Apply sanitization if available
        if self.feature_sanitizer:
            df_transformed = self.feature_sanitizer.sanitize_dataframe(df_transformed)
        
        # Ensure same features as training
        for feature in self.feature_names:
            if feature not in df_transformed.columns:
                df_transformed[feature] = 0  # Add missing features with default value
        
        # Select and order features
        df_transformed = df_transformed[self.feature_names]
        
        return df_transformed
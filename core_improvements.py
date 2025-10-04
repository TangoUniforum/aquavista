"""
Core Improvements Module for AquaVista v6.0
==========================================
Implements critical improvements for optimization and consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import pickle
import time
import gc
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from functools import wraps
import logging

# Add new imports for multicollinearity analysis
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("[WARNING] statsmodels not available. VIF calculation will be disabled.")

from scipy.stats import pearsonr
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# 1. MEMORY OPTIMIZER
class MemoryOptimizer:
    """Optimize DataFrame memory usage"""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Reduce memory usage by optimizing data types"""
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        df_optimized = df.copy()
        
        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=['int']).columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype(np.uint32)
            else:
                if col_min > -128 and col_max < 127:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
        
        # Optimize float columns
        for col in df_optimized.select_dtypes(include=['float']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Convert string columns with low cardinality to category
        for col in df_optimized.select_dtypes(include=['object']).columns:
            num_unique_values = len(df_optimized[col].unique())
            num_total_values = len(df_optimized[col])
            if num_unique_values / num_total_values < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        end_mem = df_optimized.memory_usage(deep=True).sum() / 1024**2
        
        if verbose:
            logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                       f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
        
        return df_optimized

# 2. SMART CACHE
class SmartCache:
    """Intelligent caching for expensive operations"""
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            # Import here to avoid circular imports
            try:
                from modules.config import PortableConfig
                cache_dir = str(PortableConfig.get_cache_dir())
            except ImportError:
                cache_dir = "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'saved_time': 0
        }
    
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments"""
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cache_result(self, expire_hours: int = 24):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._get_cache_key(func.__name__, args, kwargs)
                
                # Check memory cache first
                if cache_key in self.memory_cache:
                    self.cache_stats['hits'] += 1
                    logger.debug(f"Cache hit for {func.__name__}")
                    return self.memory_cache[cache_key]
                
                # Check disk cache
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    # Check if expired
                    age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                    if age_hours < expire_hours:
                        try:
                            with open(cache_file, 'rb') as f:
                                result = pickle.load(f)
                            self.memory_cache[cache_key] = result
                            self.cache_stats['hits'] += 1
                            logger.debug(f"Disk cache hit for {func.__name__}")
                            return result
                        except Exception as e:
                            logger.warning(f"Cache load failed: {e}")
                
                # Cache miss - compute result
                self.cache_stats['misses'] += 1
                start_time = time.time()
                result = func(*args, **kwargs)
                compute_time = time.time() - start_time
                
                # Save to cache
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                    self.memory_cache[cache_key] = result
                except Exception as e:
                    logger.warning(f"Cache save failed: {e}")
                
                logger.debug(f"Cache miss for {func.__name__} - computed in {compute_time:.2f}s")
                
                return result
            return wrapper
        return decorator

# 3. FEATURE NAME SANITIZER
class FeatureNameSanitizer:
    """Sanitize feature names for XGBoost, LightGBM, and other models"""
    
    def __init__(self):
        self.original_to_sanitized = {}
        self.sanitized_to_original = {}
        self.problematic_chars = re.compile(r'[<>\[\]{}.,;:|!@#$%^&*()=+/\\?`~\'"\\s-]')
    
    def sanitize_name(self, name: str) -> str:
        """Convert a single feature name to safe format"""
        # Convert to string if not already
        name = str(name)
        
        # Replace problematic characters with underscores
        sanitized = self.problematic_chars.sub('_', name)
        
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure name starts with letter or underscore (XGBoost requirement)
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = 'f_' + sanitized
        
        # Handle empty names
        if not sanitized:
            sanitized = 'feature'
        
        # Ensure uniqueness by adding suffix if needed
        base_sanitized = sanitized
        counter = 1
        while sanitized in self.sanitized_to_original and self.sanitized_to_original[sanitized] != name:
            sanitized = f"{base_sanitized}_{counter}"
            counter += 1
        
        return sanitized
    
    def sanitize_features(self, feature_names: List[str]) -> List[str]:
        """Sanitize a list of feature names and store mapping"""
        sanitized_names = []
        
        for name in feature_names:
            sanitized = self.sanitize_name(name)
            
            # Store the mapping
            self.original_to_sanitized[name] = sanitized
            self.sanitized_to_original[sanitized] = name
            
            sanitized_names.append(sanitized)
        
        # Log changes
        changed_names = [(orig, san) for orig, san in self.original_to_sanitized.items() if orig != san]
        if changed_names:
            logger.info(f"Sanitized {len(changed_names)} feature names for model compatibility:")
            for orig, san in changed_names[:5]:  # Show first 5 examples
                logger.info(f"  '{orig}' -> '{san}'")
            if len(changed_names) > 5:
                logger.info(f"  ... and {len(changed_names) - 5} more")
        
        return sanitized_names
    
    def sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize column names in a DataFrame"""
        sanitized_columns = self.sanitize_features(df.columns.tolist())
        df_sanitized = df.copy()
        df_sanitized.columns = sanitized_columns
        return df_sanitized
    
    def get_original_name(self, sanitized_name: str) -> str:
        """Get original name from sanitized name"""
        return self.sanitized_to_original.get(sanitized_name, sanitized_name)
    
    def get_sanitized_name(self, original_name: str) -> str:
        """Get sanitized name from original name"""
        return self.original_to_sanitized.get(original_name, original_name)
    
    def restore_feature_importance(self, feature_importance: Dict[str, float]) -> Dict[str, float]:
        """Convert sanitized feature importance back to original names"""
        restored = {}
        for sanitized_name, importance in feature_importance.items():
            original_name = self.get_original_name(sanitized_name)
            restored[original_name] = importance
        return restored
    
    def needs_sanitization(self, feature_names: List[str]) -> bool:
        """Check if any feature names need sanitization"""
        return any(self.problematic_chars.search(str(name)) for name in feature_names)

# 4. UNIFIED PIPELINE
class UnifiedPipeline:
    """Unified preprocessing and modeling pipeline"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.numeric_features_ = None
        self.categorical_features_ = None
        self.sanitizer = FeatureNameSanitizer()  # Add sanitizer
    
    def create_preprocessor(self, X: pd.DataFrame, 
                           numeric_features: List[str] = None,
                           categorical_features: List[str] = None) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        
        # Auto-detect feature types if not provided
        if numeric_features is None:
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if categorical_features is None:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.numeric_features_ = numeric_features
        self.categorical_features_ = categorical_features
        self.feature_names_in_ = X.columns.tolist()
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop any other columns
        )
        
        return self.preprocessor
    
    def create_model_pipeline(self, model, model_name: str, 
                            scaling_override: str = None) -> Pipeline:
        """Create complete pipeline for a model"""
        steps = []
        
        # Add preprocessor if available
        if self.preprocessor is not None:
            steps.append(('preprocessor', self.preprocessor))
        
        # Add model-specific scaling if needed
        if scaling_override:
            if scaling_override == 'minmax':
                steps.append(('model_scaler', MinMaxScaler()))
            elif scaling_override == 'robust':
                steps.append(('model_scaler', RobustScaler()))
        elif model_name in ['Neural Network', 'MLP', 'SVM']:
            # These models typically need MinMax scaling
            steps.append(('model_scaler', MinMaxScaler()))
        
        # Add the model
        steps.append(('model', model))
        
        return Pipeline(steps)
    
    def get_feature_names_out(self) -> List[str]:
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            return self.feature_names_in_
        
        feature_names = []
        
        # Get numeric feature names
        feature_names.extend(self.numeric_features_)
        
        # Get categorical feature names after one-hot encoding
        if self.categorical_features_:
            cat_transformer = self.preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'named_steps'):
                encoder = cat_transformer.named_steps['onehot']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_names = encoder.get_feature_names_out(self.categorical_features_)
                    feature_names.extend(cat_names)
        
        self.feature_names_out_ = feature_names
        return feature_names

# 5. CLASS IMBALANCE HANDLER
def handle_class_imbalance(X: pd.DataFrame, y: pd.Series, 
                          strategy: str = 'auto') -> Tuple[pd.DataFrame, pd.Series, Optional[dict]]:
    """Handle imbalanced classification datasets"""
    from collections import Counter
    
    # Check class distribution
    class_counts = Counter(y)
    total_samples = len(y)
    
    # Calculate imbalance ratio
    min_class_count = min(class_counts.values())
    max_class_count = max(class_counts.values())
    imbalance_ratio = max_class_count / min_class_count
    
    logger.info(f"Class distribution: {dict(class_counts)}")
    logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio < 3:
        logger.info("Dataset is relatively balanced")
        return X, y, None
    
    # For now, return class weights (SMOTE requires additional dependency)
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    logger.info(f"Using class weights: {class_weights}")
    
    return X, y, class_weights

# 6. TARGET DECODER
class TargetEncoder:
    """Handle target encoding and decoding"""
    
    def __init__(self):
        self.encoder = None
        self.classes_ = None
        
    def fit_transform(self, y: pd.Series) -> pd.Series:
        """Encode target variable"""
        from sklearn.preprocessing import LabelEncoder
        
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.encoder = LabelEncoder()
            y_encoded = self.encoder.fit_transform(y)
            self.classes_ = self.encoder.classes_
            logger.info(f"Encoded target classes: {list(self.classes_)}")
            return pd.Series(y_encoded, index=y.index)
        else:
            self.classes_ = None
            return y
    
    def inverse_transform(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode predictions back to original labels"""
        if self.encoder is not None:
            return self.encoder.inverse_transform(y_encoded.astype(int))
        return y_encoded
    
    def decode_proba(self, y_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """Decode probability predictions to class names"""
        if self.classes_ is not None:
            return {
                str(class_label): y_proba[:, i] 
                for i, class_label in enumerate(self.classes_)
            }
        return {"class_" + str(i): y_proba[:, i] for i in range(y_proba.shape[1])}

# 7. MULTICOLLINEARITY HANDLER
class MulticollinearityHandler:
    """Detect and handle multicollinearity in features"""
    
    def __init__(self, vif_threshold: float = 10.0, correlation_threshold: float = 0.9):
        self.vif_threshold = vif_threshold
        self.correlation_threshold = correlation_threshold
        self.vif_scores = {}
        self.high_vif_features = []
        self.correlated_pairs = []
        self.recommended_action = None
        self.analysis_results = {}
        
    def calculate_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate VIF scores for numerical features"""
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available - VIF calculation skipped")
            return {}
            
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_features) < 2:
            logger.info("Less than 2 numerical features - VIF calculation skipped")
            return {}
            
        try:
            X_numeric = X[numerical_features].dropna()
            
            if len(X_numeric) < 10:  # Need sufficient samples
                logger.warning("Insufficient samples for reliable VIF calculation")
                return {}
            
            # Add constant for VIF calculation
            X_with_const = sm.add_constant(X_numeric)
            
            vif_data = {}
            for i, feature in enumerate(numerical_features):
                try:
                    vif_score = variance_inflation_factor(X_with_const.values, i + 1)  # +1 for constant
                    
                    # Handle infinite or very large VIF values
                    if np.isinf(vif_score) or vif_score > 1000:
                        vif_score = 1000  # Cap at 1000 for display
                        
                    vif_data[feature] = vif_score
                except Exception as e:
                    logger.warning(f"Could not calculate VIF for {feature}: {e}")
                    vif_data[feature] = np.nan
                    
            self.vif_scores = vif_data
            self.high_vif_features = [feat for feat, vif in vif_data.items() 
                                     if not np.isnan(vif) and vif > self.vif_threshold]
            
            logger.info(f"VIF calculation complete. {len(self.high_vif_features)} features have VIF > {self.vif_threshold}")
            return vif_data
            
        except Exception as e:
            logger.error(f"VIF calculation failed: {e}")
            return {}
    
    def find_high_correlations(self, X: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs"""
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_features) < 2:
            return []
            
        try:
            corr_matrix = X[numerical_features].corr()
            
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    
                    if not np.isnan(corr_val) and abs(corr_val) > self.correlation_threshold:
                        feat1 = corr_matrix.columns[i]
                        feat2 = corr_matrix.columns[j]
                        high_corr_pairs.append((feat1, feat2, corr_val))
            
            self.correlated_pairs = high_corr_pairs
            logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
            return high_corr_pairs
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return []
    
    def analyze_multicollinearity(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive multicollinearity analysis"""
        logger.info("Starting multicollinearity analysis...")
        
        vif_scores = self.calculate_vif(X)
        correlated_pairs = self.find_high_correlations(X)
        
        # Determine severity
        high_vif_count = len(self.high_vif_features)
        high_corr_count = len(correlated_pairs)
        
        # Calculate severity based on both VIF and correlations
        if high_vif_count == 0 and high_corr_count == 0:
            severity = "low"
            self.recommended_action = "standard"
        elif high_vif_count <= 2 and high_corr_count <= 3:
            severity = "moderate" 
            self.recommended_action = "regularization"
        else:
            severity = "high"
            self.recommended_action = "feature_reduction"
        
        # Create comprehensive analysis results
        self.analysis_results = {
            'severity': severity,
            'vif_scores': vif_scores,
            'high_vif_features': self.high_vif_features,
            'correlated_pairs': correlated_pairs,
            'recommended_action': self.recommended_action,
            'summary': {
                'total_features': len(X.select_dtypes(include=[np.number]).columns),
                'high_vif_count': high_vif_count,
                'high_correlation_pairs': high_corr_count,
                'max_vif': max(vif_scores.values()) if vif_scores else 0,
                'max_correlation': max([abs(corr) for _, _, corr in correlated_pairs]) if correlated_pairs else 0
            },
            'recommendations': self._generate_recommendations(severity, high_vif_count, high_corr_count)
        }
        
        logger.info(f"Multicollinearity analysis complete. Severity: {severity}")
        return self.analysis_results
    
    def _generate_recommendations(self, severity: str, high_vif_count: int, high_corr_count: int) -> List[str]:
        """Generate specific recommendations based on analysis"""
        recommendations = []
        
        if severity == "low":
            recommendations.extend([
                "No significant multicollinearity detected",
                "All model types should perform well",
                "Standard feature importance interpretation is reliable"
            ])
        elif severity == "moderate":
            recommendations.extend([
                "Moderate multicollinearity detected",
                "Consider using Ridge or ElasticNet regression",
                "Tree-based models (Random Forest, XGBoost) should handle this well",
                "Monitor model stability across different train/test splits"
            ])
        else:  # high
            recommendations.extend([
                "High multicollinearity detected - action recommended",
                "Use regularized models (Ridge, Lasso, ElasticNet)",
                "Consider removing features with VIF > 10",
                "PCA or feature clustering may help",
                "Be cautious with feature importance interpretation"
            ])
            
            if high_vif_count > 0:
                recommendations.append(f"Remove or combine {high_vif_count} high-VIF features")
            
            if high_corr_count > 5:
                recommendations.append("Consider feature clustering to reduce redundancy")
        
        return recommendations
    
    def get_adaptive_model_recommendations(self, severity: str) -> List[str]:
        """Get model recommendations based on multicollinearity severity"""
        if severity == "low":
            return ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting", "Linear Regression"]
        elif severity == "moderate":
            return ["Ridge", "ElasticNet", "Random Forest", "Bayesian Ridge", "XGBoost"]
        else:  # high
            return ["Ridge", "Lasso", "ElasticNet", "Bayesian Ridge", "PCA + Linear"]
    
    def remove_redundant_features(self, X: pd.DataFrame, method: str = "vif") -> Tuple[pd.DataFrame, List[str]]:
        """Remove redundant features based on VIF or correlation"""
        X_reduced = X.copy()
        removed_features = []
        
        if method == "vif" and STATSMODELS_AVAILABLE:
            # Iteratively remove highest VIF features
            max_iterations = min(len(X.columns) // 2, 10)  # Limit iterations
            
            for iteration in range(max_iterations):
                vif_scores = self.calculate_vif(X_reduced)
                if not vif_scores:
                    break
                    
                max_vif_feature = max(vif_scores, key=vif_scores.get)
                max_vif_value = vif_scores[max_vif_feature]
                
                if max_vif_value > self.vif_threshold:
                    X_reduced = X_reduced.drop(columns=[max_vif_feature])
                    removed_features.append(max_vif_feature)
                    logger.info(f"Removed {max_vif_feature} (VIF: {max_vif_value:.2f})")
                else:
                    break
                    
        elif method == "correlation":
            # Remove one feature from each highly correlated pair
            for feat1, feat2, corr_val in self.correlated_pairs:
                if feat1 in X_reduced.columns and feat2 in X_reduced.columns:
                    # Remove the feature with higher mean correlation to other features
                    numerical_cols = X_reduced.select_dtypes(include=[np.number]).columns
                    if len(numerical_cols) > 1:
                        corr_matrix = X_reduced[numerical_cols].corr()
                        
                        if feat1 in corr_matrix.columns and feat2 in corr_matrix.columns:
                            feat1_mean_corr = corr_matrix[feat1].abs().mean()
                            feat2_mean_corr = corr_matrix[feat2].abs().mean()
                            
                            if feat1_mean_corr > feat2_mean_corr:
                                X_reduced = X_reduced.drop(columns=[feat1])
                                removed_features.append(feat1)
                                logger.info(f"Removed {feat1} (high correlation with {feat2}: {corr_val:.3f})")
                            else:
                                X_reduced = X_reduced.drop(columns=[feat2])
                                removed_features.append(feat2)
                                logger.info(f"Removed {feat2} (high correlation with {feat1}: {corr_val:.3f})")
        
        logger.info(f"Feature reduction complete. Removed {len(removed_features)} features due to multicollinearity")
        return X_reduced, removed_features
    
    def create_pca_features(self, X: pd.DataFrame, n_components: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """Create PCA features to handle multicollinearity"""
        try:
            numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_features) < 2:
                logger.warning("Insufficient numerical features for PCA")
                return X, None
            
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X[numerical_features])
            
            # Create PCA feature names
            pca_feature_names = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
            
            # Create new dataframe with PCA features
            X_pca_df = pd.DataFrame(X_pca, columns=pca_feature_names, index=X.index)
            
            # Add back non-numerical features
            non_numerical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numerical_features:
                X_final = pd.concat([X_pca_df, X[non_numerical_features]], axis=1)
            else:
                X_final = X_pca_df
            
            logger.info(f"PCA applied: {len(numerical_features)} features -> {X_pca.shape[1]} components")
            logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
            
            return X_final, pca
            
        except Exception as e:
            logger.error(f"PCA transformation failed: {e}")
            return X, None
    
    def get_treatment_summary(self) -> Dict[str, Any]:
        """Get summary of multicollinearity treatment applied"""
        return {
            'analysis_performed': bool(self.analysis_results),
            'severity': self.analysis_results.get('severity', 'unknown'),
            'features_flagged': len(self.high_vif_features),
            'correlations_found': len(self.correlated_pairs),
            'recommended_action': self.recommended_action,
            'vif_available': STATSMODELS_AVAILABLE
        }

# Initialize module-level cache instance
cache = SmartCache()

# Convenience function for sanitization
def sanitize_feature_names(feature_names: List[str]) -> Tuple[List[str], FeatureNameSanitizer]:
    """Convenience function to sanitize feature names and return sanitizer"""
    sanitizer = FeatureNameSanitizer()
    sanitized_names = sanitizer.sanitize_features(feature_names)
    return sanitized_names, sanitizer

# Convenience function for multicollinearity analysis
def analyze_multicollinearity(X: pd.DataFrame, vif_threshold: float = 10.0, 
                            correlation_threshold: float = 0.9) -> Tuple[Dict[str, Any], MulticollinearityHandler]:
    """Convenience function to analyze multicollinearity and return handler"""
    handler = MulticollinearityHandler(vif_threshold, correlation_threshold)
    analysis = handler.analyze_multicollinearity(X)
    return analysis, handler
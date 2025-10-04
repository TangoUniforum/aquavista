"""
Model Manager Module for AquaVista v7.0
=======================================
Handles model training, hyperparameter optimization, and ensemble creation.
Now includes Bayesian models with uncertainty quantification and enhanced SHAP handling.
FIXED: Scaling issues resolved to prevent data corruption and inconsistent application.
ENHANCED: Added comprehensive debugging for SHAP inconsistencies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.base import clone
from sklearn.model_selection import (cross_val_score, GridSearchCV, RandomizedSearchCV,
                                   learning_curve, validation_curve)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, mean_squared_error, mean_absolute_error,
                           r2_score, explained_variance_score)
from sklearn.ensemble import (VotingClassifier, VotingRegressor, 
                            StackingClassifier, StackingRegressor)
import time
import warnings
import logging
import joblib
from pathlib import Path
import psutil
from datetime import datetime

# Import all sklearn models
from sklearn.linear_model import (LinearRegression, LogisticRegression, Ridge, Lasso,
                                ElasticNet, SGDRegressor, SGDClassifier, RidgeClassifier,
                                BayesianRidge, ARDRegression, HuberRegressor,
                                RANSACRegressor, TheilSenRegressor,
                                PassiveAggressiveRegressor, PassiveAggressiveClassifier)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                            GradientBoostingRegressor, GradientBoostingClassifier,
                            AdaBoostRegressor, AdaBoostClassifier,
                            ExtraTreesRegressor, ExtraTreesClassifier,
                            BaggingRegressor, BaggingClassifier,
                            HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

# Optional advanced models
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Import Bayesian models (v7.0)
try:
    from modules.bayesian_models import (
        BayesianLinearRegression,
        BayesianLogisticRegression,
        BayesianRidge as CustomBayesianRidge,  # Renamed to avoid conflict with sklearn
        GaussianProcessRegressor as CustomGaussianProcess,
        plot_posterior_predictive_check
    )
    BAYESIAN_MODELS_AVAILABLE = True
except ImportError:
    BAYESIAN_MODELS_AVAILABLE = False
    print("[WARNING] Bayesian models module not available. Install pymc and arviz to enable.")

# Import custom modules with error handling
try:
    from modules.config import Config, ModelTrainingError
except ImportError:
    print("[ERROR] Could not import Config. Check modules.config module.")
    raise

try:
    from modules.interpretability import InterpretabilityEngine
except ImportError:
    print("[ERROR] Could not import InterpretabilityEngine. Check modules.interpretability module.")
    raise

try:
    from modules.core_improvements import FeatureNameSanitizer
except ImportError:
    print("[WARNING] Could not import FeatureNameSanitizer. Some features may not work.")
    # Create fallback
    class FeatureNameSanitizer:
        def needs_sanitization(self, names): return False
        def sanitize_dataframe(self, df): return df
        def restore_feature_importance(self, imp): return imp
        def get_original_name(self, name): return name

import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid duplicate handlers in Streamlit
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    ch = logging.StreamHandler(stream=sys.stdout)
    try:
        ch.stream.reconfigure(encoding="utf-8")  # Python 3.9+
    except AttributeError:
        pass  # Fallback silently if not supported
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

warnings.filterwarnings('ignore')


class ModelManager:
    """Manages model training, optimization, and evaluation with enhanced SHAP support and FIXED scaling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.model_definitions = self._initialize_model_definitions()
        self.interpretability_engine = InterpretabilityEngine(config)
        self.feature_sanitizer = None
    
    def _initialize_model_definitions(self) -> Dict[str, Dict]:
        """Initialize model definitions with metadata"""
        definitions = {
            # Linear Models
            'Linear Regression': {
                'class': LinearRegression,
                'type': 'regression',
                'category': 'Linear',
                'params': {},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': True
            },
            'Logistic Regression': {
                'class': LogisticRegression,
                'type': 'classification',
                'category': 'Linear',
                'params': {'max_iter': 1000},
                'supports_multiclass': True,
                'feature_importance': 'coef_',
                'scales_well': True
            },
            'Ridge': {
                'class': Ridge,
                'type': 'regression',
                'category': 'Linear',
                'params': {},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': True
            },
            'Ridge Classifier': {
                'class': RidgeClassifier,
                'type': 'classification',
                'category': 'Linear',
                'params': {},
                'supports_multiclass': True,
                'feature_importance': 'coef_',
                'scales_well': True
            },
            'Lasso': {
                'class': Lasso,
                'type': 'regression',
                'category': 'Linear',
                'params': {'max_iter': 2000},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': True
            },
            'ElasticNet': {
                'class': ElasticNet,
                'type': 'regression',
                'category': 'Linear',
                'params': {'max_iter': 2000},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': True
            },
            'SGD Regressor': {
                'class': SGDRegressor,
                'type': 'regression',
                'category': 'Linear',
                'params': {'max_iter': 1000, 'tol': 1e-3},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': True,
                'requires_scaling': True
            },
            'Passive Aggressive': {
                'class': PassiveAggressiveRegressor,
                'type': 'regression',
                'category': 'Linear',
                'params': {'max_iter': 1000, 'tol': 1e-3},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': True,
                'requires_scaling': True
            },
            
            # Tree Models
            'Decision Tree': {
                'class_regression': DecisionTreeRegressor,
                'class_classification': DecisionTreeClassifier,
                'type': 'both',
                'category': 'Tree',
                'params': {},
                'supports_multiclass': True,
                'feature_importance': 'feature_importances_',
                'scales_well': True
            },
            'Random Forest': {
                'class_regression': RandomForestRegressor,
                'class_classification': RandomForestClassifier,
                'type': 'both',
                'category': 'Tree',
                'params': {'n_jobs': -1},
                'supports_multiclass': True,
                'feature_importance': 'feature_importances_',
                'scales_well': True
            },
            'Extra Trees': {
                'class_regression': ExtraTreesRegressor,
                'class_classification': ExtraTreesClassifier,
                'type': 'both',
                'category': 'Tree',
                'params': {'n_jobs': -1},
                'supports_multiclass': True,
                'feature_importance': 'feature_importances_',
                'scales_well': True
            },
            'Bagging': {
                'class_regression': BaggingRegressor,
                'class_classification': BaggingClassifier,
                'type': 'both',
                'category': 'Tree',
                'params': {'n_jobs': -1},
                'supports_multiclass': True,
                'feature_importance': None,  # Bagging doesn't have reliable feature importance
                'scales_well': True
            },
            
            # Boosting Models
            'Gradient Boosting': {
                'class_regression': GradientBoostingRegressor,
                'class_classification': GradientBoostingClassifier,
                'type': 'both',
                'category': 'Boosting',
                'params': {},
                'supports_multiclass': True,
                'feature_importance': 'feature_importances_',
                'scales_well': False
            },
            'Hist Gradient Boosting': {
                'class_regression': HistGradientBoostingRegressor,
                'class_classification': HistGradientBoostingClassifier,
                'type': 'both',
                'category': 'Boosting',
                'params': {},
                'supports_multiclass': True,
                'feature_importance': None,
                'scales_well': True
            },
            'AdaBoost': {
                'class_regression': AdaBoostRegressor,
                'class_classification': AdaBoostClassifier,
                'type': 'both',
                'category': 'Boosting',
                'params': {},
                'supports_multiclass': False,
                'feature_importance': 'feature_importances_',
                'scales_well': False
            },
            
            # Support Vector Machines
            'SVM': {
                'class_regression': SVR,
                'class_classification': SVC,
                'type': 'both',
                'category': 'SVM',
                'params': {},
                'supports_multiclass': True,
                'feature_importance': None,
                'scales_well': False
            },
            
            # Neighbors
            'K-Neighbors': {
                'class_regression': KNeighborsRegressor,
                'class_classification': KNeighborsClassifier,
                'type': 'both',
                'category': 'Neighbors',
                'params': {'n_jobs': -1},
                'supports_multiclass': True,
                'feature_importance': None,
                'scales_well': False
            },
            
            # Neural Networks
            'Neural Network': {
                'class_regression': MLPRegressor,
                'class_classification': MLPClassifier,
                'type': 'both',
                'category': 'Neural',
                'params': {'max_iter': 1000, 'early_stopping': True},
                'supports_multiclass': True,
                'feature_importance': None,
                'scales_well': True,
                'requires_scaling': True
            },
            
            # Bayesian Models
            'Naive Bayes': {
                'class': GaussianNB,
                'type': 'classification',
                'category': 'Bayesian',
                'params': {},
                'supports_multiclass': True,
                'feature_importance': None,
                'scales_well': True
            },
            'Bayesian Ridge': {
                'class': BayesianRidge,
                'type': 'regression',
                'category': 'Bayesian',
                'params': {},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': True
            },
            'ARD Regression': {
                'class': ARDRegression,
                'type': 'regression',
                'category': 'Bayesian',
                'params': {'max_iter': 300},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': True
            },
            
            # Robust Regressors
            'Huber': {
                'class': HuberRegressor,
                'type': 'regression',
                'category': 'Robust',
                'params': {},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': True
            },
            'RANSAC': {
                'class': RANSACRegressor,
                'type': 'regression',
                'category': 'Robust',
                'params': {},
                'supports_multiclass': False,
                'feature_importance': None,
                'scales_well': False
            },
            'Theil-Sen': {
                'class': TheilSenRegressor,
                'type': 'regression',
                'category': 'Robust',
                'params': {'max_iter': 300, 'n_jobs': -1},
                'supports_multiclass': False,
                'feature_importance': 'coef_',
                'scales_well': False,
                'small_data': True
            },
            
            # Gaussian Process
            'Gaussian Process': {
                'class_regression': GaussianProcessRegressor,
                'class_classification': GaussianProcessClassifier,
                'type': 'both',
                'category': 'Gaussian Process',
                'params': {},
                'supports_multiclass': True,
                'feature_importance': None,
                'scales_well': False,
                'small_data': True
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            definitions['XGBoost'] = {
                'class_regression': XGBRegressor,
                'class_classification': XGBClassifier,
                'type': 'both',
                'category': 'Boosting',
                'params': {
                    'n_jobs': -1,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                },
                'supports_multiclass': True,
                'feature_importance': 'feature_importances_',
                'scales_well': True
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            definitions['LightGBM'] = {
                'class_regression': LGBMRegressor,
                'class_classification': LGBMClassifier,
                'type': 'both',
                'category': 'Boosting',
                'params': {'n_jobs': -1, 'verbose': -1},
                'supports_multiclass': True,
                'feature_importance': 'feature_importances_',
                'scales_well': True
            }
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            definitions['CatBoost'] = {
                'class_regression': CatBoostRegressor,
                'class_classification': CatBoostClassifier,
                'type': 'both',
                'category': 'Boosting',
                'params': {'verbose': False},
                'supports_multiclass': True,
                'feature_importance': 'feature_importances_',
                'scales_well': True
            }
        
        # Add v7.0 Bayesian models if available
        if BAYESIAN_MODELS_AVAILABLE:
            definitions.update({
                "Bayesian Linear Regression": {
                    "class": BayesianLinearRegression,
                    "type": "regression",
                    "params": {
                        "n_samples": [1000, 2000, 4000],
                        "n_chains": [2, 4],
                        "target_accept": [0.8, 0.9, 0.95]
                    },
                    "supports_proba": True,
                    "supports_uncertainty": True,
                    "supports_multiclass": False,
                    "feature_importance": None,
                    "category": "Bayesian Models"
                },
                "Bayesian Ridge Regression": {
                    "class": CustomBayesianRidge,
                    "type": "regression",
                    "params": {
                        "n_samples": [1000, 2000],
                        "n_chains": [2, 4],
                        "ard": [True, False]
                    },
                    "supports_proba": False,
                    "supports_uncertainty": True,
                    "supports_feature_importance": True,
                    "supports_multiclass": False,
                    "feature_importance": None,
                    "category": "Bayesian Models"
                },
                "Gaussian Process Regression": {
                    "class": CustomGaussianProcess,
                    "type": "regression",
                    "params": {
                        "kernel": ["rbf", "matern32"],
                        "n_samples": [500, 1000],
                        "n_chains": [2]
                    },
                    "supports_proba": False,
                    "supports_uncertainty": True,
                    "supports_multiclass": False,
                    "feature_importance": None,
                    "category": "Bayesian Models"
                },
                "Bayesian Logistic Regression": {
                    "class": BayesianLogisticRegression,
                    "type": "classification",
                    "params": {
                        "n_samples": [1000, 2000],
                        "n_chains": [2, 4],
                        "target_accept": [0.8, 0.9]
                    },
                    "supports_proba": True,
                    "supports_uncertainty": True,
                    "supports_multiclass": False,  # Only binary
                    "feature_importance": None,
                    "category": "Bayesian Models"
                }
            })
        
        return definitions
    
    def get_available_models_count(self) -> int:
        """Get count of available models"""
        return len(self.model_definitions)
    
    def get_available_models(self, task_type: str, n_samples: int = None) -> Dict[str, List[str]]:
        """Get available models for task type grouped by category"""
        available = {}
        
        # First, get n_classes if it's stored in processed_data
        n_classes = None
        if hasattr(self, 'n_classes'):
            n_classes = self.n_classes
        
        for model_name, model_def in self.model_definitions.items():
            # Check if model supports task type
            if model_def['type'] == 'both' or model_def['type'] == task_type:
                category = model_def['category']
                if category not in available:
                    available[category] = []
                available[category].append(model_name)
        
        # Add Bayesian models if available
        if BAYESIAN_MODELS_AVAILABLE:
            bayesian_models = {
                "regression": [
                    "Bayesian Linear Regression",
                    "Bayesian Ridge Regression",
                    "Gaussian Process Regression"
                ],
                "classification": [
                    "Bayesian Logistic Regression"
                ]
            }
            
            if task_type == "regression":
                if "Bayesian Models" not in available:
                    available["Bayesian Models"] = []
                for model in bayesian_models["regression"]:
                    if model in self.model_definitions and model not in available["Bayesian Models"]:
                        available["Bayesian Models"].append(model)
            elif task_type == "classification":
                # Check if binary classification for Bayesian Logistic
                if n_classes == 2 or n_classes is None:  # Allow if n_classes unknown
                    if "Bayesian Models" not in available:
                        available["Bayesian Models"] = []
                    for model in bayesian_models["classification"]:
                        if model in self.model_definitions and model not in available["Bayesian Models"]:
                            available["Bayesian Models"].append(model)
        
        return available
    
    def get_model_preset(self, preset: str, available_models: Dict[str, List[str]], 
                        task_type: str) -> List[str]:
        """Get model selection based on preset"""
        all_models = [model for models in available_models.values() for model in models]
        
        if "Recommended" in preset:
            if task_type == 'regression':
                recommended = ['Random Forest', 'XGBoost', 'LightGBM', 'Ridge', 
                             'Neural Network', 'Gradient Boosting', 'ARD Regression',
                             'Bayesian Linear Regression', 'Bayesian Ridge Regression']
            else:
                recommended = ['Random Forest', 'XGBoost', 'LightGBM', 
                             'Logistic Regression', 'Neural Network', 'Gradient Boosting',
                             'Bayesian Logistic Regression']
            return [m for m in recommended if m in all_models]
        
        elif preset == "⚡ Fast Models":
            fast_models = ['Ridge', 'Logistic Regression', 'Naive Bayes', 
                          'K-Neighbors', 'Decision Tree', 'SGD Regressor']
            return [m for m in fast_models if m in all_models][:6]
        
        elif preset == "🏆 High Accuracy":
            accurate_models = ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest',
                             'Gradient Boosting', 'Extra Trees', 'Neural Network',
                             'Bagging', 'Stacking Ensemble', 'Bayesian Ridge Regression',
                             'Gaussian Process Regression']
            return [m for m in accurate_models if m in all_models]
        
        elif preset == "🔬 All Models":
            return all_models
        
        return []
    
    def estimate_memory_usage(self, selected_models: List[str], n_samples: int, 
                            n_features: int) -> float:
        """Estimate memory usage for selected models in MB"""
        base_memory = n_samples * n_features * 8 / (1024 * 1024)  # Basic dataset
        
        model_memory = 0
        for model_name in selected_models:
            model_def = self.model_definitions.get(model_name, {})
            
            # Estimate based on model type
            if model_def.get('category') == 'Tree':
                model_memory += base_memory * 2  # Trees store data
            elif model_def.get('category') == 'Boosting':
                model_memory += base_memory * 3  # Multiple trees
            elif model_def.get('category') == 'Neural':
                model_memory += base_memory * 1.5  # Weights
            elif model_def.get('category') == 'Robust' and model_name == 'Theil-Sen':
                model_memory += base_memory * 4  # Computationally intensive
            elif model_def.get('category') == 'Bayesian Models':
                model_memory += base_memory * 5  # MCMC samples storage
            else:
                model_memory += base_memory * 0.5  # Linear models
        
        return base_memory + model_memory
    
    def create_model_instance(self, model_name: str, task_type: str) -> Any:
        """Create model instance"""
        model_def = self.model_definitions.get(model_name)
        
        if not model_def:
            raise ModelTrainingError(f"Unknown model: {model_name}")
        
        # Get model class
        if model_def['type'] == 'both':
            if task_type == 'regression':
                model_class = model_def['class_regression']
            else:
                model_class = model_def['class_classification']
        else:
            model_class = model_def['class']
        
        # Create instance with default params
        params = model_def.get('params', {}).copy()

        # XGBoost needs an explicit eval_metric
        if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            if task_type == 'classification':
                params.setdefault('eval_metric', 'logloss')
            else:  # regression
                params.setdefault('eval_metric', 'rmse')

        # Special handling for specific models
        if model_name == 'SVM' and task_type == 'classification':
            params['probability'] = True  # Only for classification
        
        # For v7.0 Bayesian models, use default params (not grid search params)
        if model_def.get('category') == 'Bayesian Models':
            # These models have grid search params, so we need defaults
            if model_name == 'Bayesian Linear Regression':
                params = {'n_samples': 2000, 'n_chains': 4, 'target_accept': 0.9}
            elif model_name == 'Bayesian Ridge Regression':
                params = {'n_samples': 2000, 'n_chains': 4, 'ard': True}
            elif model_name == 'Gaussian Process Regression':
                params = {'kernel': 'rbf', 'n_samples': 1000, 'n_chains': 2}
            elif model_name == 'Bayesian Logistic Regression':
                params = {'n_samples': 2000, 'n_chains': 4, 'target_accept': 0.9}
        
        # Add random state if applicable
        if 'random_state' in model_class().get_params():
            params['random_state'] = self.config.computation.random_state
        
        return model_class(**params)
    
    def _is_already_scaled(self, X_data: pd.DataFrame) -> bool:
        """FIXED: Check if data appears to already be scaled to prevent double-scaling"""
        numerical_cols = X_data.select_dtypes(include=[np.number])
        if len(numerical_cols.columns) == 0:
            return False
        
        # Check if features have mean near 0 and std near 1 (standardized)
        # or values in [0,1] range (min-max scaled)
        means = numerical_cols.mean()
        stds = numerical_cols.std()
        mins = numerical_cols.min()
        maxs = numerical_cols.max()
        
        # Check for standard scaling (mean ~0, std ~1)
        near_zero_mean = (abs(means) < 0.2).sum() / len(means) > 0.7
        near_one_std = (abs(stds - 1) < 0.3).sum() / len(stds) > 0.7
        standardized = near_zero_mean and near_one_std
        
        # Check for min-max scaling (values in [0,1])
        in_unit_range = ((mins >= -0.1) & (maxs <= 1.1)).sum() / len(numerical_cols.columns) > 0.8
        
        return standardized or in_unit_range
    
    def _get_model_specific_scaler(self, model_name: str, X_train, scaling_method: str = 'auto'):
        """FIXED: Determine the best scaler for a specific model with Linear Regression included"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        if scaling_method != 'auto' and scaling_method != 'auto (best for each model)':
            # User specified a specific scaling method
            if scaling_method == 'standard':
                return StandardScaler()
            elif scaling_method == 'minmax':
                return MinMaxScaler()
            elif scaling_method == 'robust':
                return RobustScaler()
            else:
                return None
        
        # Auto mode - choose best scaler for each model
        
        # Models that require specific scaling
        if model_name in ['Neural Network', 'SVM', 'K-Neighbors']:
            # These models work better with MinMax scaling [0,1]
            return MinMaxScaler()
        
        elif model_name in ['SGD Regressor', 'Passive Aggressive']:
            # These always need standardization
            return StandardScaler()
        
        # FIXED: Added Linear Regression to the list
        elif model_name in ['Ridge', 'Lasso', 'ElasticNet', 'Logistic Regression', 'Linear Regression']:
            # Linear models work well with standardization
            return StandardScaler()
        
        elif model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 
                           'LightGBM', 'CatBoost', 'Extra Trees', 'AdaBoost', 'Bagging',
                           'Hist Gradient Boosting']:
            # Tree-based models don't need scaling
            return None
        
        elif model_name in ['ARD Regression', 'Bayesian Ridge', 'Huber']:
            # Bayesian and robust models work well with standardization
            return StandardScaler()
        
        elif model_name == 'Theil-Sen':
            # Theil-Sen benefits from robust scaling
            return RobustScaler()
        
        # v7.0 Bayesian models
        elif model_name in ['Bayesian Linear Regression', 'Bayesian Ridge Regression', 
                           'Bayesian Logistic Regression']:
            # These models handle scaling internally
            return None
        
        elif model_name == 'Gaussian Process Regression':
            # GP benefits from standardization
            return StandardScaler()
        
        else:
            # For other models, check for outliers
            numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_columns) == 0:
                return None
                
            outlier_ratio = 0
            for col in numerical_columns[:10]:  # Check first 10 columns
                Q1 = X_train[col].quantile(0.25)
                Q3 = X_train[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = ((X_train[col] < Q1 - 1.5*IQR) | (X_train[col] > Q3 + 1.5*IQR)).sum()
                    outlier_ratio += outliers / len(X_train)
            
            outlier_ratio /= min(10, len(numerical_columns))
            
            # Use robust scaler if many outliers
            if outlier_ratio > 0.1:
                return RobustScaler()
            else:
                return StandardScaler()

    def get_param_grid(self, model_name: str, tuning_budget: str = 'standard') -> Dict:
        """Get hyperparameter grid for model"""
        # Define comprehensive parameter grids
        param_grids = {
            'Linear Regression': {},
            
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100] if tuning_budget == 'extensive' else [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            
            'Ridge': {
                'alpha': [0.01, 0.1, 1, 10, 100] if tuning_budget == 'extensive' else [0.1, 1, 10]
            },
            
            'Lasso': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1] if tuning_budget == 'extensive' else [0.001, 0.01, 0.1]
            },
            
            'ElasticNet': {
                'alpha': [0.001, 0.01, 0.1] if tuning_budget != 'quick' else [0.01],
                'l1_ratio': [0.1, 0.5, 0.9] if tuning_budget != 'quick' else [0.5]
            },
            
            'SGD Regressor': {
                'loss': ['squared_error', 'huber', 'epsilon_insensitive'] if tuning_budget != 'quick' else ['squared_error'],
                'penalty': ['l2', 'l1', 'elasticnet'] if tuning_budget == 'extensive' else ['l2'],
                'alpha': [0.0001, 0.001, 0.01] if tuning_budget != 'quick' else [0.0001],
                'learning_rate': ['invscaling', 'optimal', 'adaptive'] if tuning_budget != 'quick' else ['invscaling'],
                'eta0': [0.01, 0.1] if tuning_budget == 'extensive' else [0.01]
            },
            
            'Passive Aggressive': {
                'C': [0.01, 0.1, 1.0, 10.0] if tuning_budget != 'quick' else [1.0],
                'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                'epsilon': [0.01, 0.1, 0.5] if tuning_budget != 'quick' else [0.1],
                'fit_intercept': [True, False] if tuning_budget == 'extensive' else [True]
            },
            
            'Decision Tree': {
                'max_depth': [3, 5, 10, 15, None] if tuning_budget == 'extensive' else [5, 10, None],
                'min_samples_split': [2, 5, 10] if tuning_budget != 'quick' else [2, 5],
                'min_samples_leaf': [1, 2, 4] if tuning_budget != 'quick' else [1, 2]
            },
            
            'Random Forest': {
                'n_estimators': [100, 200, 300] if tuning_budget == 'extensive' else [100, 200],
                'max_depth': [10, 20, None] if tuning_budget != 'quick' else [None],
                'min_samples_split': [2, 5] if tuning_budget != 'quick' else [2],
                'max_features': ['sqrt', 'log2'] if tuning_budget != 'quick' else ['sqrt']
            },
            
            'Bagging': {
                'n_estimators': [10, 50, 100] if tuning_budget != 'quick' else [50],
                'max_samples': [0.5, 0.7, 1.0] if tuning_budget != 'quick' else [1.0],
                'max_features': [0.5, 0.7, 1.0] if tuning_budget != 'quick' else [1.0],
                'bootstrap': [True, False] if tuning_budget == 'extensive' else [True],
                'bootstrap_features': [False, True] if tuning_budget == 'extensive' else [False]
            },
            
            'Gradient Boosting': {
                'n_estimators': [100, 200] if tuning_budget != 'quick' else [100],
                'learning_rate': [0.01, 0.1, 0.3] if tuning_budget == 'extensive' else [0.1],
                'max_depth': [3, 5, 7] if tuning_budget != 'quick' else [3, 5],
                'subsample': [0.8, 1.0] if tuning_budget != 'quick' else [1.0]
            },
            
            'XGBoost': {
                'n_estimators': [100, 200, 300] if tuning_budget == 'extensive' else [100, 200],
                'learning_rate': [0.01, 0.1, 0.3] if tuning_budget != 'quick' else [0.1],
                'max_depth': [3, 6, 9] if tuning_budget != 'quick' else [3, 6],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0] if tuning_budget != 'quick' else [1.0]
            },
            
            'LightGBM': {
                'n_estimators': [100, 200] if tuning_budget != 'quick' else [100],
                'learning_rate': [0.01, 0.1] if tuning_budget != 'quick' else [0.1],
                'num_leaves': [31, 50, 100] if tuning_budget == 'extensive' else [31, 50],
                'feature_fraction': [0.8, 1.0] if tuning_budget != 'quick' else [1.0]
            },
            
            'SVM': {
                'C': [0.1, 1, 10] if tuning_budget != 'quick' else [1],
                'kernel': ['rbf', 'poly'] if tuning_budget == 'extensive' else ['rbf'],
                'gamma': ['scale', 'auto']
            },
            
            'K-Neighbors': {
                'n_neighbors': [3, 5, 7, 9] if tuning_budget != 'quick' else [5],
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski', 'manhattan'] if tuning_budget == 'extensive' else ['minkowski']
            },
            
            'Neural Network': {
                'hidden_layer_sizes': [(100,), (100, 50), (100, 100)] if tuning_budget != 'quick' else [(100,)],
                'activation': ['relu', 'tanh'] if tuning_budget == 'extensive' else ['relu'],
                'alpha': [0.0001, 0.001, 0.01] if tuning_budget != 'quick' else [0.001],
                'learning_rate': ['constant', 'adaptive'] if tuning_budget != 'quick' else ['adaptive']
            },
            
            'ARD Regression': {
                'alpha_1': [1e-6, 1e-5, 1e-4] if tuning_budget != 'quick' else [1e-6],
                'alpha_2': [1e-6, 1e-5, 1e-4] if tuning_budget != 'quick' else [1e-6],
                'lambda_1': [1e-6, 1e-5, 1e-4] if tuning_budget != 'quick' else [1e-6],
                'lambda_2': [1e-6, 1e-5, 1e-4] if tuning_budget != 'quick' else [1e-6],
                'threshold_lambda': [1000, 10000, 100000] if tuning_budget == 'extensive' else [10000]
            },
            
            'Bayesian Ridge': {
                'alpha_1': [1e-6, 1e-5, 1e-4] if tuning_budget != 'quick' else [1e-6],
                'alpha_2': [1e-6, 1e-5, 1e-4] if tuning_budget != 'quick' else [1e-6],
                'lambda_1': [1e-6, 1e-5, 1e-4] if tuning_budget != 'quick' else [1e-6],
                'lambda_2': [1e-6, 1e-5, 1e-4] if tuning_budget != 'quick' else [1e-6]
            },
            
            'Huber': {
                'epsilon': [1.1, 1.35, 1.5, 2.0] if tuning_budget != 'quick' else [1.35],
                'alpha': [0.0001, 0.001, 0.01] if tuning_budget != 'quick' else [0.0001]
            },
            
            'Theil-Sen': {
                'max_subpopulation': [10000, 100000] if tuning_budget != 'quick' else [10000],
                'n_subsamples': [None] if tuning_budget == 'quick' else [None, 50, 100],
                'fit_intercept': [True]
            },
            
            # v7.0 Bayesian models
            'Bayesian Linear Regression': {
                'n_samples': [1000, 2000] if tuning_budget != 'quick' else [1000],
                'n_chains': [2, 4] if tuning_budget != 'quick' else [2],
                'target_accept': [0.8, 0.9] if tuning_budget != 'quick' else [0.9]
            },
            
            'Bayesian Ridge Regression': {
                'n_samples': [1000, 2000] if tuning_budget != 'quick' else [1000],
                'n_chains': [2, 4] if tuning_budget != 'quick' else [2],
                'ard': [True, False] if tuning_budget != 'quick' else [True]
            },
            
            'Gaussian Process Regression': {
                'kernel': ['rbf', 'matern32'] if tuning_budget != 'quick' else ['rbf'],
                'n_samples': [500, 1000] if tuning_budget != 'quick' else [500],
                'n_chains': [2]
            },
            
            'Bayesian Logistic Regression': {
                'n_samples': [1000, 2000] if tuning_budget != 'quick' else [1000],
                'n_chains': [2, 4] if tuning_budget != 'quick' else [2],
                'target_accept': [0.8, 0.9] if tuning_budget != 'quick' else [0.9]
            }
        }
        
        # Return grid or empty dict
        return param_grids.get(model_name, {})
    
    def _optimize_hyperparameters(self, model, param_grid: Dict, X_train, y_train,
                                task_type: str, cv_folds: int, search_method: str,
                                n_iter: Optional[int]) -> Tuple[Any, Dict]:
        """ENHANCED: Optimize hyperparameters using grid or random search with fixed random states"""
        scoring = 'r2' if task_type == 'regression' else 'accuracy'
        
        if search_method == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv_folds, scoring=scoring,
                n_jobs=self.config.computation.n_jobs, verbose=0,
                random_state=self.config.computation.random_state  # FIXED: Added random state
            )
        else:
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv_folds,
                scoring=scoring, n_jobs=self.config.computation.n_jobs,
                verbose=0, 
                random_state=self.config.computation.random_state  # FIXED: Added random state
            )
        
        # Log hyperparameter search details
        logger.info(f"Starting hyperparameter optimization with {search_method} search")
        logger.info(f"Parameter grid: {param_grid}")
        
        search.fit(X_train, y_train)
        
        # Enhanced logging of results
        logger.info(f"Hyperparameter optimization complete:")
        logger.info(f"  Best score: {search.best_score_:.6f}")
        logger.info(f"  Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.cv_results_
    
    def _calculate_scores(self, y_true, y_pred, task_type: str, 
                         y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive scores"""
        scores = {}
        
        if task_type == 'regression':
            scores['r2_score'] = r2_score(y_true, y_pred)
            scores['mse'] = mean_squared_error(y_true, y_pred)
            scores['rmse'] = np.sqrt(scores['mse'])
            scores['mae'] = mean_absolute_error(y_true, y_pred)
            scores['explained_variance'] = explained_variance_score(y_true, y_pred)
            
            # MAPE
            mask = y_true != 0
            if mask.any():
                scores['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            
        else:  # classification
            scores['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Handle multiclass
            n_classes = len(np.unique(y_true))
            average = 'weighted' if n_classes > 2 else 'binary'
            
            scores['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
            scores['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
            scores['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
            
            # AUC for binary classification
            if n_classes == 2 and y_proba is not None:
                scores['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        
        return scores
    
    def debug_model_training(self, X_train, y_train, X_test, y_test, model, model_name):
        """ENHANCED: Debug what the model actually learns vs expected correlations"""
        
        logger.info(f"\n=== ENHANCED MODEL TRAINING DEBUG: {model_name} ===")
        
        # Check input correlations
        if hasattr(X_train, 'corrwith'):
            input_correlations = X_train.corrwith(y_train).sort_values(ascending=False)
            logger.info("Top 10 input correlations:")
            for feat, corr in input_correlations.head(10).items():
                logger.info(f"  {feat}: {corr:+.6f}")
            logger.info("Bottom 5 input correlations:")
            for feat, corr in input_correlations.tail(5).items():
                logger.info(f"  {feat}: {corr:+.6f}")
        else:
            logger.info("Cannot compute input correlations - no corrwith method")
            input_correlations = None
        
        # Check model predictions distribution
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        logger.info(f"Prediction statistics:")
        logger.info(f"  Train predictions: mean={train_pred.mean():.6f}, std={train_pred.std():.6f}")
        logger.info(f"  Test predictions: mean={test_pred.mean():.6f}, std={test_pred.std():.6f}")
        logger.info(f"  Train targets: mean={y_train.mean():.6f}, std={y_train.std():.6f}")
        logger.info(f"  Test targets: mean={y_test.mean():.6f}, std={y_test.std():.6f}")
        
        # Check prediction correlation with targets
        train_corr = np.corrcoef(y_train, train_pred)[0, 1]
        test_corr = np.corrcoef(y_test, test_pred)[0, 1]
        logger.info(f"Prediction correlations:")
        logger.info(f"  Train correlation: {train_corr:+.6f}")
        logger.info(f"  Test correlation: {test_corr:+.6f}")
        
        # Check feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(model.feature_importances_, index=X_train.columns)
            logger.info("Model feature importance (top 10):")
            for feat, imp in importance.nlargest(10).items():
                input_corr = input_correlations.get(feat, 0) if input_correlations is not None else 0
                logger.info(f"  {feat}: importance={imp:.6f}, input_corr={input_corr:+.6f}")
        
        # Check linear model coefficients
        if hasattr(model, 'coef_'):
            coefs = pd.Series(model.coef_.flatten(), index=X_train.columns)
            logger.info("Model coefficients vs input correlations (top 10):")
            for feat in coefs.abs().nlargest(10).index:
                if feat in coefs and (input_correlations is None or feat in input_correlations):
                    coef = coefs[feat]
                    corr = input_correlations.get(feat, 0) if input_correlations is not None else 0
                    sign_match = (coef > 0) == (corr > 0) if corr != 0 else "N/A"
                    status = "MATCH" if sign_match == True else ("MISMATCH" if sign_match == False else "N/A")
                    logger.info(f"  {feat}: coef={coef:+.6f}, corr={corr:+.6f} [{status}]")
        
        # Check for key suspicious features mentioned in the issue
        suspicious_features = ['square_feet', 'quare_feet', 'crime_rate', 'distance_to_city', 'di_tance_to_city']
        found_features = [f for f in suspicious_features if f in X_train.columns]
        
        if found_features:
            logger.info("Analysis of potentially problematic features:")
            for feat in found_features:
                # Basic stats
                feat_mean = X_train[feat].mean()
                feat_std = X_train[feat].std()
                feat_min = X_train[feat].min()
                feat_max = X_train[feat].max()
                
                logger.info(f"  {feat}: mean={feat_mean:.6f}, std={feat_std:.6f}, range=[{feat_min:.6f}, {feat_max:.6f}]")
                
                # Correlation with target
                if input_correlations is not None and feat in input_correlations:
                    logger.info(f"    Target correlation: {input_correlations[feat]:+.6f}")
                
                # Model's learned relationship
                if hasattr(model, 'feature_importances_') and feat in X_train.columns:
                    idx = X_train.columns.get_loc(feat)
                    importance = model.feature_importances_[idx]
                    logger.info(f"    Model importance: {importance:.6f}")
                
                if hasattr(model, 'coef_') and feat in X_train.columns:
                    idx = X_train.columns.get_loc(feat)
                    coef = model.coef_.flatten()[idx]
                    logger.info(f"    Model coefficient: {coef:+.6f}")
        
        logger.info(f"=== END ENHANCED MODEL TRAINING DEBUG: {model_name} ===\n")
    
    def _train_single_model(self, model_name: str, X_train, X_test, y_train, y_test,
                          task_type: str, use_cv: bool, cv_folds: int,
                          optimize_hyperparams: bool, tuning_budget: str,
                          scaling_method: str = 'auto',
                          numerical_features: List[str] = None) -> Dict:
        """ENHANCED: Train a single model with comprehensive debugging for SHAP consistency issues"""
        start_time = time.time()
        
        logger.info(f"\n>>> Starting training for {model_name} <<<")
        logger.info(f"Hyperparameter optimization: {optimize_hyperparams}")
        logger.info(f"Tuning budget: {tuning_budget}")
        logger.info(f"Scaling method: {scaling_method}")
        
        # Get model definition
        model_def = self.model_definitions[model_name]
        
        # Create model instance
        model = self.create_model_instance(model_name, task_type)
        
        # Log initial model parameters
        logger.info(f"Initial model parameters: {model.get_params()}")
        
        # Special handling for specific models
        if model_name == 'RANSAC':
            # Set min_samples based on training data size
            n_samples = len(X_train)
            min_samples = min(X_train.shape[1] + 1, n_samples // 2)
            model.min_samples = min_samples
        
        elif model_name == 'Theil-Sen':
            # Adjust for large datasets
            n_samples = len(X_train)
            if n_samples > 5000:
                model.max_subpopulation = min(1000, n_samples // 10)
        
        # FIXED: Get model-specific scaler with improved logic
        scaler = self._get_model_specific_scaler(model_name, X_train, scaling_method)
        
        # FIXED: Apply scaling with double-scaling prevention and correlation validation
        if scaler is not None:
            # FIXED: Check if data is already scaled to prevent double-scaling
            if self._is_already_scaled(X_train):
                logger.warning(f"Data appears already scaled - skipping additional scaling for {model_name}")
                X_train_scaled = X_train
                X_test_scaled = X_test
                scaler = None
            else:
                # Apply scaling
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                logger.info(f"Applied {type(scaler).__name__} for {model_name}")
                
                # CRITICAL: Validate scaling didn't corrupt correlations
                if hasattr(X_train, 'corrwith'):
                    original_corr = X_train.corrwith(y_train)
                    scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                    scaled_corr = scaled_df.corrwith(y_train)
                    
                    # Check key features for correlation preservation
                    for feature in ['square_feet', 'quare_feet', 'crime_rate', 'distance_to_city', 'di_tance_to_city']:
                        if feature in original_corr and feature in scaled_corr:
                            orig = original_corr[feature]
                            scaled = scaled_corr[feature]
                            corr_change = abs(orig - scaled)
                            
                            if corr_change > 0.1:  # Significant correlation change indicates corruption
                                logger.error(f"🚨 SCALING CORRUPTION DETECTED: {feature} correlation changed from {orig:.6f} to {scaled:.6f}")
                                logger.error(f"   Correlation change: {corr_change:.6f} - This indicates data corruption!")
                                logger.error(f"   Reverting to unscaled data for {model_name}")
                                
                                # Revert to unscaled data to prevent corruption
                                X_train_scaled = X_train
                                X_test_scaled = X_test
                                scaler = None
                                break
                            else:
                                logger.info(f"   Correlation preserved: {feature} {orig:.6f} -> {scaled:.6f} (Δ={corr_change:.6f})")
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            logger.info(f"No scaling applied for {model_name}")
        
        # Convert to DataFrame if numpy arrays (for feature name handling)
        if isinstance(X_train_scaled, np.ndarray):
            original_columns = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train_scaled.shape[1])]
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=original_columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=original_columns)
        
        # Special handling for tree-based models that need sanitized features
        needs_sanitization = model_name in ['XGBoost', 'LightGBM', 'CatBoost']
        feature_sanitizer = None
        
        if needs_sanitization:
            # Use sanitizer from processed_data if available, otherwise create new one
            if self.feature_sanitizer:
                feature_sanitizer = self.feature_sanitizer
                # Feature names should already be sanitized from data processing
                logger.info(f"Using existing feature sanitizer for {model_name}")
            else:
                feature_sanitizer = FeatureNameSanitizer()
                if feature_sanitizer.needs_sanitization(X_train_scaled.columns.tolist()):
                    logger.info(f"Creating new feature sanitizer for {model_name}")
                    X_train_scaled = feature_sanitizer.sanitize_dataframe(X_train_scaled)
                    X_test_scaled = feature_sanitizer.sanitize_dataframe(X_test_scaled)
                else:
                    feature_sanitizer = None
        
        # Optimize hyperparameters
        cv_results = None
        if optimize_hyperparams:
            param_grid = self.get_param_grid(model_name, tuning_budget)
            
            if param_grid:
                logger.info(f"Starting hyperparameter optimization for {model_name}")
                logger.info(f"Parameter grid size: {np.prod([len(v) if isinstance(v, list) else 1 for v in param_grid.values()])}")
                
                if tuning_budget == 'quick':
                    search_method = 'grid'
                    n_iter = None
                else:
                    search_method = 'random'
                    n_iter = 20 if tuning_budget == 'standard' else 50
                
                model, cv_results = self._optimize_hyperparameters(
                    model, param_grid, X_train_scaled, y_train,
                    task_type, cv_folds, search_method, n_iter
                )
                
                logger.info(f"Hyperparameter optimization complete for {model_name}")
                logger.info(f"Best parameters: {model.get_params()}")
                if cv_results is not None:
                    logger.info(f"Best CV score: {cv_results['mean_test_score'].max():.6f}")
            else:
                logger.info(f"No hyperparameter grid defined for {model_name} - using default parameters")
        else:
            logger.info(f"Hyperparameter optimization disabled for {model_name}")
        
        # Store pre-fit model parameters for comparison
        pre_fit_params = model.get_params().copy()
        
        # Fit final model
        logger.info(f"Fitting final model: {model_name}")
        model.fit(X_train_scaled, y_train)
        
        # Store post-fit parameters
        post_fit_params = model.get_params().copy()
        
        # Compare parameters before and after fit
        param_changes = {k: (pre_fit_params.get(k), post_fit_params.get(k)) 
                        for k in set(pre_fit_params.keys()) | set(post_fit_params.keys()) 
                        if pre_fit_params.get(k) != post_fit_params.get(k)}
        
        if param_changes:
            logger.info(f"Parameters changed during fit: {param_changes}")
        
        # ENHANCED DEBUG BLOCK - inserted right after model.fit()
        logger.info(f"=== COMPREHENSIVE MODEL ANALYSIS: {model_name} ===")
        logger.info(f"Hyperparameter optimization enabled: {optimize_hyperparams}")
        logger.info(f"Final model parameters: {model.get_params()}")
        
        # Make predictions for debugging
        train_pred_debug = model.predict(X_train_scaled)
        test_pred_debug = model.predict(X_test_scaled)
        
        # Log prediction statistics
        logger.info(f"Prediction Analysis:")
        logger.info(f"  Training predictions - mean: {train_pred_debug.mean():.6f}, std: {train_pred_debug.std():.6f}")
        logger.info(f"  Test predictions - mean: {test_pred_debug.mean():.6f}, std: {test_pred_debug.std():.6f}")
        logger.info(f"  Training targets - mean: {y_train.mean():.6f}, std: {y_train.std():.6f}")
        logger.info(f"  Test targets - mean: {y_test.mean():.6f}, std: {y_test.std():.6f}")
        
        # Correlation analysis
        if len(train_pred_debug) > 1 and len(test_pred_debug) > 1:
            train_pred_corr = np.corrcoef(y_train, train_pred_debug)[0, 1]
            test_pred_corr = np.corrcoef(y_test, test_pred_debug)[0, 1]
            logger.info(f"  Prediction-target correlations: train={train_pred_corr:+.6f}, test={test_pred_corr:+.6f}")
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(model.feature_importances_, index=X_train_scaled.columns)
            logger.info("Top 10 Feature Importances:")
            for feat, imp in importance.nlargest(10).items():
                logger.info(f"  {feat}: {imp:.6f}")
        
        # Coefficient analysis for linear models
        if hasattr(model, 'coef_'):
            coef_series = pd.Series(model.coef_.flatten(), index=X_train_scaled.columns)
            logger.info("Top 5 Positive Coefficients:")
            for feat, coef in coef_series.nlargest(5).items():
                logger.info(f"  {feat}: {coef:+.6f}")
            logger.info("Top 5 Negative Coefficients:")
            for feat, coef in coef_series.nsmallest(5).items():
                logger.info(f"  {feat}: {coef:+.6f}")
        
        # Run the comprehensive debug analysis
        if logger.isEnabledFor(logging.INFO):
            self.debug_model_training(X_train_scaled, y_train, X_test_scaled, y_test, model, model_name)
        
        logger.info(f"=== END COMPREHENSIVE MODEL ANALYSIS: {model_name} ===")
        
        # Make final predictions
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, 'predict_proba') and task_type == 'classification':
            y_proba = model.predict_proba(X_test_scaled)
        else:
            y_proba = None
        
        # Calculate scores
        train_scores = self._calculate_scores(
            y_train, model.predict(X_train_scaled), task_type, y_proba=None
        )
        test_scores = self._calculate_scores(y_test, y_pred, task_type, y_proba)
        
        logger.info(f"Final Scores for {model_name}:")
        logger.info(f"  Train scores: {train_scores}")
        logger.info(f"  Test scores: {test_scores}")
        
        # Cross-validation scores
        if use_cv:
            cv_scores = self._get_cv_scores(
                model, X_train_scaled, y_train, task_type, cv_folds
            )
            logger.info(f"  CV scores: {cv_scores}")
        else:
            cv_scores = None
        
        # Feature importance
        feature_importance = self._get_feature_importance(
            model, model_def, X_train_scaled.columns.tolist(), feature_sanitizer
        )
        
        # Learning curve
        if cv_folds > 2 and len(X_train) > 50:
            learning_curve_data = self._get_learning_curve(
                model, X_train_scaled, y_train, task_type, cv_folds
            )
        else:
            learning_curve_data = None
        
        training_time = time.time() - start_time
        
        logger.info(f"Training completed for {model_name} in {training_time:.2f} seconds")
        
        # Build the result dictionary
        result = {
            'model': model,
            'model_name': model_name,
            'scaler': scaler,
            'scaler_type': type(scaler).__name__ if scaler else None,
            'feature_sanitizer': feature_sanitizer,
            'best_params': model.get_params() if hasattr(model, 'get_params') else {},
            'pre_fit_params': pre_fit_params,  # DEBUG: Store pre-fit parameters
            'post_fit_params': post_fit_params,  # DEBUG: Store post-fit parameters
            'param_changes': param_changes,  # DEBUG: Store parameter changes
            'hyperparameter_optimized': optimize_hyperparams,  # DEBUG: Flag for optimization
            'train_scores': train_scores,
            'test_scores': test_scores,
            'cv_scores': cv_scores,
            'cv_results': cv_results,
            'predictions': y_pred,
            'probabilities': y_proba,
            'feature_importance': feature_importance,
            'learning_curve': learning_curve_data,
            'training_time': training_time,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Check if it's a Bayesian model (v7.0)
        if model_def.get("category") == "Bayesian Models":
            # Store additional Bayesian-specific information
            result["is_bayesian"] = True
            result["supports_uncertainty"] = model_def.get("supports_uncertainty", False)
            
            # Get posterior diagnostics if available
            if hasattr(model, 'trace_'):
                result["posterior_diagnostics"] = {
                    "n_samples": model.n_samples,
                    "n_chains": model.n_chains,
                    "convergence": self._check_convergence(model.trace_)
                }
        
        return result
    
    def _get_cv_scores(self, model, X, y, task_type: str, cv_folds: int) -> Dict[str, Any]:
        """Get cross-validation scores"""
        scoring = 'r2' if task_type == 'regression' else 'accuracy'
        
        scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring=scoring,
            n_jobs=self.config.computation.n_jobs
        )
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'scores': scores.tolist(),
            'cv_folds': cv_folds
        }
    
    def _get_feature_importance(self, model, model_def: Dict, 
                              feature_names: List[str], 
                              sanitizer: FeatureNameSanitizer = None) -> Dict[str, Any]:
        """Extract feature importance from model"""
        importance_data = {
            'feature_names': feature_names,
            'importances': {}
        }
        
        # Get importance attribute
        importance_attr = model_def.get('feature_importance')
        
        if importance_attr and hasattr(model, importance_attr):
            importances = getattr(model, importance_attr)
            
            # Debug logging
            logger.debug(f"Feature importance extraction for {model.__class__.__name__}:")
            logger.debug(f"Number of features: {len(feature_names)}")
            logger.debug(f"Shape of importances: {importances.shape if hasattr(importances, 'shape') else 'N/A'}")
            
            if hasattr(importances, 'shape'):
                if len(importances.shape) == 1:
                    # Single dimensional importance array
                    if len(importances) == len(feature_names):
                        for i, (feature, importance) in enumerate(zip(feature_names, importances)):
                            clean_feature = str(feature).strip()
                            importance_data['importances'][clean_feature] = float(importance)
                    else:
                        logger.warning(f"Mismatch: {len(feature_names)} features but {len(importances)} importances")
                        min_len = min(len(feature_names), len(importances))
                        for i in range(min_len):
                            clean_feature = str(feature_names[i]).strip()
                            importance_data['importances'][clean_feature] = float(importances[i])
                else:
                    # Multi-dimensional (multi-class) - average across classes
                    avg_importances = np.mean(np.abs(importances), axis=0)
                    if len(avg_importances) == len(feature_names):
                        for i, (feature, importance) in enumerate(zip(feature_names, avg_importances)):
                            clean_feature = str(feature).strip()
                            importance_data['importances'][clean_feature] = float(importance)
                    else:
                        logger.warning(f"Mismatch: {len(feature_names)} features but {len(avg_importances)} importances")
                        min_len = min(len(feature_names), len(avg_importances))
                        for i in range(min_len):
                            clean_feature = str(feature_names[i]).strip()
                            importance_data['importances'][clean_feature] = float(avg_importances[i])
        
        # Special handling for v7.0 Bayesian Ridge with ARD
        elif model_def.get('category') == 'Bayesian Models' and hasattr(model, 'get_feature_importance'):
            try:
                importances = model.get_feature_importance()
                for i, (feature, importance) in enumerate(zip(feature_names, importances)):
                    clean_feature = str(feature).strip()
                    importance_data['importances'][clean_feature] = float(importance)
            except Exception as e:
                logger.warning(f"Could not get feature importance for {model.__class__.__name__}: {e}")
        
        # Restore original feature names if sanitization was used
        if sanitizer and importance_data['importances']:
            importance_data['importances'] = sanitizer.restore_feature_importance(
                importance_data['importances']
            )
            importance_data['feature_names'] = [
                sanitizer.get_original_name(name) for name in feature_names
            ]
        
        return importance_data
    
    def _get_learning_curve(self, model, X, y, task_type: str, cv_folds: int) -> Dict[str, Any]:
        """Calculate learning curve data"""
        scoring = 'r2' if task_type == 'regression' else 'accuracy'
        
        # Adjust train sizes based on dataset size
        n_samples = len(X)
        if n_samples < 100:
            train_sizes = np.linspace(0.3, 1.0, 5)
        else:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv_folds, scoring=scoring,
            n_jobs=self.config.computation.n_jobs,
            train_sizes=train_sizes
        )
        
        return {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
            'val_scores_std': np.std(val_scores, axis=1).tolist()
        }
    
    def _check_convergence(self, trace):
        """Check MCMC convergence using R-hat"""
        try:
            import arviz as az
            
            rhat = az.rhat(trace)
            max_rhat = max(float(v.max()) for v in rhat.data_vars.values())
            
            return {
                "converged": max_rhat < 1.01,
                "max_rhat": max_rhat,
                "warning": "Consider increasing n_samples or n_chains" if max_rhat >= 1.01 else None
            }
        except Exception as e:
            logger.warning(f"Could not check convergence: {e}")
            return {
                "converged": None,
                "max_rhat": None,
                "warning": "Could not compute convergence diagnostics"
            }
    
    def _calculate_shap_with_fallbacks(self, model, model_name: str, X_test, task_type: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Calculate SHAP values with comprehensive fallbacks for problematic models"""
        
        diagnostics = {
            'model_name': model_name,
            'attempts': [],
            'final_method': None,
            'success': False,
            'error': None
        }
        
        # Model-specific configurations
        model_configs = {
            'SVM': {
                'force_kernel': True,
                'background_samples': 25,  # Very small background for SVM
                'max_samples': 50,         # Small sample size
                'use_linear_approx': True
            },
            'K-Neighbors': {
                'background_samples': 50,
                'max_samples': 100,
                'use_kernel_only': True
            },
            'Neural Network': {
                'background_samples': 100,
                'max_samples': 200,
                'try_deep_explainer': True
            },
            'Gaussian Process': {
                'background_samples': 25,   # GP is expensive
                'max_samples': 50
            },
            'Bagging': {
                'background_samples': 50,
                'max_samples': 100,
                'use_kernel_only': True     # Bagging doesn't have feature_importances reliably
            },
            'Hist Gradient Boosting': {
                'background_samples': 75,
                'max_samples': 150,
                'use_kernel_only': True    # HistGradientBoosting doesn't support TreeExplainer
            },
            'AdaBoost': {
                'background_samples': 75,
                'max_samples': 150,
                'prefer_kernel': True      # AdaBoost TreeExplainer can be unstable
            },
            'Naive Bayes': {
                'background_samples': 50,
                'max_samples': 100,
                'use_kernel_only': True
            }
        }
        
        config = model_configs.get(model_name, {
            'background_samples': 100,
            'max_samples': 500
        })
        
        # Limit sample sizes
        max_samples = min(config.get('max_samples', 500), len(X_test))
        X_sample = X_test.sample(n=max_samples, random_state=42) if len(X_test) > max_samples else X_test.copy()
        
        # Attempt 1: Use the enhanced interpretability engine
        try:
            diagnostics['attempts'].append('Enhanced InterpretabilityEngine')
            shap_values = self.interpretability_engine.calculate_shap_values(
                model, X_sample, task_type, background_samples=config.get('background_samples', 100)
            )
            
            if shap_values is not None:
                diagnostics['final_method'] = 'Enhanced InterpretabilityEngine'
                diagnostics['success'] = True
                return shap_values, diagnostics
                
        except Exception as e:
            diagnostics['attempts'].append(f'Enhanced InterpretabilityEngine failed: {str(e)}')
            logger.debug(f"Enhanced SHAP failed for {model_name}: {e}")
        
        # Attempt 2: Model-specific fallback strategies
        try:
            import shap
            
            if model_name == 'SVM':
                diagnostics['attempts'].append('SVM-specific KernelExplainer')
                # For SVM, use very small background and sample
                background = shap.sample(X_test, min(25, len(X_test)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample.head(25), nsamples=50)  # Very conservative
                
            elif model_name == 'K-Neighbors':
                diagnostics['attempts'].append('K-Neighbors KernelExplainer')
                # K-NN benefits from larger background but still limited
                background = shap.sample(X_test, min(50, len(X_test)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample.head(50), nsamples=100)
                
            elif model_name == 'Neural Network':
                diagnostics['attempts'].append('Neural Network specialized approach')
                # Try different approaches for neural networks
                if hasattr(model, 'predict_proba') and task_type == 'classification':
                    background = shap.sample(X_test, min(100, len(X_test)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X_sample.head(100))
                else:
                    background = shap.sample(X_test, min(100, len(X_test)))
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(X_sample.head(100), nsamples=200)
                    
            elif model_name == 'Gaussian Process':
                diagnostics['attempts'].append('Gaussian Process minimal approach')
                # GP is computationally expensive, use minimal samples
                background = shap.sample(X_test, min(10, len(X_test)))  # Very small
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample.head(25), nsamples=30)  # Minimal
                
            elif model_name in ['Bagging', 'Hist Gradient Boosting', 'Naive Bayes']:
                diagnostics['attempts'].append(f'{model_name} KernelExplainer')
                # These don't always have reliable TreeExplainer support
                background = shap.sample(X_test, min(75, len(X_test)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample.head(100), nsamples=150)
                
            elif model_name == 'AdaBoost':
                diagnostics['attempts'].append('AdaBoost KernelExplainer fallback')
                # AdaBoost TreeExplainer can be unstable, prefer Kernel
                background = shap.sample(X_test, min(75, len(X_test)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample.head(100), nsamples=150)
                
            else:
                # Generic fallback
                diagnostics['attempts'].append('Generic KernelExplainer fallback')
                background = shap.sample(X_test, min(config.get('background_samples', 100), len(X_test)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample.head(200), nsamples='auto')
            
            if shap_values is not None:
                diagnostics['final_method'] = f'{model_name}-specific approach'
                diagnostics['success'] = True
                return shap_values, diagnostics
                
        except Exception as e:
            diagnostics['attempts'].append(f'Model-specific approach failed: {str(e)}')
            logger.debug(f"Model-specific SHAP failed for {model_name}: {e}")
        
        # Attempt 3: Ultra-conservative fallback
        try:
            import shap
            diagnostics['attempts'].append('Ultra-conservative fallback')
            
            # Minimal everything approach
            background = shap.sample(X_test, min(10, len(X_test)))
            explainer = shap.KernelExplainer(model.predict, background)
            
            # Use very small sample and iterations
            tiny_sample = X_sample.head(10)
            shap_values = explainer.shap_values(tiny_sample, nsamples=25)
            
            if shap_values is not None:
                diagnostics['final_method'] = 'Ultra-conservative fallback'
                diagnostics['success'] = True
                return shap_values, diagnostics
                
        except Exception as e:
            diagnostics['attempts'].append(f'Ultra-conservative fallback failed: {str(e)}')
            logger.debug(f"Ultra-conservative SHAP failed for {model_name}: {e}")
        
        # All attempts failed
        diagnostics['error'] = f"All SHAP calculation methods failed for {model_name}"
        logger.warning(f"All SHAP methods failed for {model_name}")
        return None, diagnostics
    
    def train_models(self, selected_models: List[str], processed_data: Dict,
                    use_cv: bool = True, cv_folds: int = 5,
                    optimize_hyperparams: bool = True, tuning_budget: str = 'standard',
                    ensemble_methods: List[str] = None,
                    progress_callback: Callable = None) -> Dict:
        """Train multiple models with optimization - ENHANCED SHAP VERSION"""
        
        results = {
            'models': {},
            'task_type': processed_data['task_type'],
            'best_model': None,
            'best_score': -np.inf,
            'baseline_score': 0,
            'primary_metric': 'r2_score' if processed_data['task_type'] == 'regression' else 'accuracy',
            'total_time': 0,
            'model_performance_summary': {}
        }
        
        # Get data
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        task_type = processed_data['task_type']
        
        # Store sanitizer for use in individual model training
        if 'feature_sanitizer' in processed_data:
            self.feature_sanitizer = processed_data['feature_sanitizer']
        else:
            self.feature_sanitizer = None
        
        # Store n_classes for get_available_models
        if task_type == 'classification':
            self.n_classes = len(np.unique(y_train))
        
        # Calculate baseline
        if task_type == 'regression':
            baseline_pred = np.full_like(y_test, y_train.mean())
            results['baseline_score'] = r2_score(y_test, baseline_pred)
        else:
            from collections import Counter
            most_common = Counter(y_train).most_common(1)[0][0]
            baseline_pred = np.full_like(y_test, most_common)
            results['baseline_score'] = accuracy_score(y_test, baseline_pred)
        
        start_time = time.time()
        
        # Initialize performance tracking
        results['model_performance_summary'] = {
            'excellent': [],
            'good': [],
            'poor': [],
            'failed': []
        }
        
        # Train each model
        for idx, model_name in enumerate(selected_models):
            try:
                if progress_callback:
                    progress = (idx + 0.5) / len(selected_models)
                    progress_callback(progress, f"Training {model_name}...")
                
                # Train model
                model_results = self._train_single_model(
                    model_name, X_train, X_test, y_train, y_test,
                    task_type, use_cv, cv_folds, optimize_hyperparams, tuning_budget,
                    scaling_method=processed_data.get('scaling_method', 'auto'),
                    numerical_features=processed_data.get('numerical_features', [])
                )
                
                results['models'][model_name] = model_results
                
                # Check if best model
                score = model_results['test_scores'][results['primary_metric']]
                if score > results['best_score']:
                    results['best_score'] = score
                    results['best_model'] = model_name
                
                # Categorize performance
                baseline = results['baseline_score']
                if score < baseline:
                    results['model_performance_summary']['poor'].append({
                        'model': model_name,
                        'score': score,
                        'vs_baseline': f"{((score - baseline) / abs(baseline) * 100):.1f}%"
                    })
                elif score >= baseline * 1.5:
                    results['model_performance_summary']['excellent'].append({
                        'model': model_name,
                        'score': score,
                        'vs_baseline': f"+{((score - baseline) / abs(baseline) * 100):.1f}%"
                    })
                else:
                    results['model_performance_summary']['good'].append({
                        'model': model_name,
                        'score': score,
                        'vs_baseline': f"+{((score - baseline) / abs(baseline) * 100):.1f}%"
                    })
                
                logger.info(f"Trained {model_name}: {results['primary_metric']}={score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                results['model_performance_summary']['failed'].append({
                    'model': model_name,
                    'error': str(e)
                })
                if progress_callback:
                    progress = (idx + 1) / len(selected_models)
                    progress_callback(progress, f"Failed to train {model_name}")
        
        # Create ensembles
        if ensemble_methods and len(results['models']) >= 2:
            if progress_callback:
                progress_callback(0.9, "Creating ensemble models...")
            
            ensemble_results = self._create_ensembles(
                results['models'], X_train, X_test, y_train, y_test,
                task_type, ensemble_methods
            )
            
            for ens_name, ens_results in ensemble_results.items():
                results['models'][ens_name] = ens_results
                
                # Check if best
                score = ens_results['test_scores'][results['primary_metric']]
                if score > results['best_score']:
                    results['best_score'] = score
                    results['best_model'] = ens_name
        
        # ROBUST SHAP CALCULATION with guaranteed storage
        if self.config.features.shap_analysis:
            if progress_callback:
                progress_callback(0.95, "Calculating SHAP values...")
            
            shap_summary = {
                'attempted': 0,
                'successful': 0,
                'failed': 0,
                'models_with_shap': []
            }
            
            # Process all models (including ensembles)
            for model_name, model_data in results['models'].items():
                shap_summary['attempted'] += 1
                
                try:
                    logger.info(f"Calculating SHAP for {model_name}...")
                    
                    # Calculate SHAP values with fallbacks
                    shap_values, shap_diagnostics = self._calculate_shap_with_fallbacks(
                        model_data['model'], model_name, X_test, task_type
                    )
                    
                    # ALWAYS store SHAP values if calculated, regardless of validation
                    if shap_values is not None:
                        model_data['shap_values'] = shap_values
                        model_data['shap_available'] = True
                        model_data['shap_diagnostics'] = shap_diagnostics
                        # Mathematical validation
                        math_valid, math_details = self.validate_shap_mathematically(
                            shap_values, model_data['model'], X_test
                        )
                        model_data['shap_math_valid'] = math_valid
                        model_data['shap_math_details'] = math_details

                        if math_valid:
                            logger.info(f"SHAP mathematical validation passed for {model_name}")
                        else:
                            logger.warning(f"SHAP mathematical validation failed for {model_name}: {math_details.get('issues', [])}")

                        # Perform validation but don't reject based on results
                        try:
                            is_valid, validation_details = self.interpretability_engine.validate_shap_values(
                                shap_values, model_name
                            )
                            model_data['shap_validated'] = is_valid
                            model_data['shap_validation_details'] = validation_details
                            
                            if is_valid:
                                logger.info(f"SHAP validation passed for {model_name}")
                            else:
                                logger.info(f"SHAP validation failed for {model_name} but values stored anyway")
                                
                        except Exception as validation_error:
                            logger.warning(f"SHAP validation error for {model_name}: {validation_error}")
                            model_data['shap_validated'] = False
                            model_data['shap_validation_error'] = str(validation_error)
                        
                        shap_summary['successful'] += 1
                        shap_summary['models_with_shap'].append(model_name)
                        
                        logger.info(f"SHAP values stored for {model_name} (method: {shap_diagnostics.get('final_method', 'unknown')})")
                        
                    else:
                        # No SHAP values calculated
                        model_data['shap_available'] = False
                        model_data['shap_diagnostics'] = shap_diagnostics
                        model_data['shap_error'] = shap_diagnostics.get('error', 'SHAP calculation failed')
                        shap_summary['failed'] += 1
                        logger.warning(f"SHAP calculation failed for {model_name}: {shap_diagnostics.get('error', 'Unknown error')}")
                
                except Exception as e:
                    logger.error(f"Unexpected error in SHAP calculation for {model_name}: {str(e)}")
                    model_data['shap_available'] = False
                    model_data['shap_error'] = str(e)
                    shap_summary['failed'] += 1
            
            # Store comprehensive SHAP summary
            results['shap_summary'] = {
                'total_models': shap_summary['attempted'],
                'successful_calculations': shap_summary['successful'],
                'failed_calculations': shap_summary['failed'],
                'success_rate': f"{(shap_summary['successful'] / max(shap_summary['attempted'], 1) * 100):.1f}%",
                'models_with_shap': shap_summary['models_with_shap'],
                'shap_enabled': True
            }
            
            logger.info(f"SHAP Summary: {shap_summary['successful']}/{shap_summary['attempted']} models have SHAP values")
        
        results['total_time'] = time.time() - start_time
        
        # Store best model data
        if results['best_model']:
            results['best_model_data'] = results['models'][results['best_model']]
        
        # Generate performance recommendations
        results['recommendations'] = self.get_model_recommendations_from_results(results)
        
        return results
    
    def get_model_recommendations_from_results(self, training_results: Dict[str, Any]) -> str:
        """Generate recommendations based on actual performance"""
        recommendations = []
        
        if not training_results.get('models'):
            return "No models trained yet."
        
        # Get performance summary
        perf_summary = training_results.get('model_performance_summary', {})
        
        # Top performers
        if perf_summary.get('excellent'):
            excellent_models = [m['model'] for m in perf_summary['excellent'][:3]]
            recommendations.append(f"Excellent performers: {', '.join(excellent_models)}")
            recommendations.append(f"   These models significantly outperform the baseline")
        
        if perf_summary.get('good'):
            good_models = [m['model'] for m in perf_summary['good'][:3]]
            recommendations.append(f"Good performers: {', '.join(good_models)}")
        
        # Poor performers
        if perf_summary.get('poor'):
            poor_models = [m['model'] for m in perf_summary['poor']]
            recommendations.append(f"Poor performers: {', '.join(poor_models)}")
            recommendations.append(f"   These models perform worse than the baseline")
        
        # Failed models
        if perf_summary.get('failed'):
            failed_models = [m['model'] for m in perf_summary['failed']]
            recommendations.append(f"Failed to train: {', '.join(failed_models)}")
        
        # Overall analysis
        total_models = len(training_results['models'])
        baseline = training_results['baseline_score']
        
        if all(model_data['test_scores'][training_results['primary_metric']] < baseline * 1.1 
               for model_data in training_results['models'].values()):
            recommendations.append("\nData Quality Suggestions:")
            recommendations.append("   • Consider feature engineering to create more informative features")
            recommendations.append("   • Check for data quality issues (outliers, noise)")
            recommendations.append("   • Ensure sufficient training data")
            recommendations.append("   • Verify correct problem formulation")
        
        # Model-specific insights
        if perf_summary.get('poor'):
            recommendations.append("\nWhy some models failed:")
            poor_model_names = [m['model'] for m in perf_summary['poor']]
            
            if 'Gaussian Process' in poor_model_names:
                recommendations.append("   • Gaussian Process: May not scale well with your data size")
            if 'SVM' in poor_model_names:
                recommendations.append("   • SVM: Try feature scaling or different kernel")
            if 'Theil-Sen' in poor_model_names:
                recommendations.append("   • Theil-Sen: Computationally intensive, may timeout on large datasets")
        
        # Excellent performer insights
        if perf_summary.get('excellent'):
            excellent_model_names = [m['model'] for m in perf_summary['excellent']]
            
            if any(model in excellent_model_names for model in ['ARD Regression', 'Bayesian Ridge']):
                recommendations.append("\nBayesian models performing well: Your data likely has clear feature relevance patterns")
            
            if any(model in excellent_model_names for model in ['Bayesian Linear Regression', 'Bayesian Ridge Regression']):
                recommendations.append("\nv7.0 Bayesian models excelling: Consider exploring uncertainty quantification for risk assessment")
            
            if 'SGD Regressor' in excellent_model_names:
                recommendations.append("SGD success: Consider online learning for real-time predictions")
        
        return "\n".join(recommendations)
    
    def _create_ensembles(self, base_models: Dict, X_train, X_test, y_train, y_test,
                         task_type: str, methods: List[str]) -> Dict[str, Dict]:
        """Create ensemble models with proper scaling handling"""
        ensemble_results = {}
    
        # Filter out failed models
        valid_models = {name: data for name, data in base_models.items() 
                       if 'model' in data and data['model'] is not None}
    
        if len(valid_models) < 2:
            return ensemble_results
    
        # CRITICAL FIX: We need to handle predictions with proper scaling
    
        # Voting ensemble
        if 'Voting' in methods:
            try:
                if task_type == 'regression':
                    # Collect predictions from each model WITH PROPER SCALING
                    train_predictions = []
                    test_predictions = []
                
                    for name, data in valid_models.items():
                        model = data['model']
                        scaler = data.get('scaler')
                    
                        # Apply the model's specific scaler if it exists
                        if scaler is not None:
                            X_train_scaled = scaler.transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            train_pred = model.predict(X_train_scaled)
                            test_pred = model.predict(X_test_scaled)
                        else:
                            # Model doesn't use scaling
                            train_pred = model.predict(X_train)
                            test_pred = model.predict(X_test)
                    
                        train_predictions.append(train_pred)
                        test_predictions.append(test_pred)
                
                    # Average the predictions (simple voting for regression)
                    train_voting_pred = np.mean(train_predictions, axis=0)
                    test_voting_pred = np.mean(test_predictions, axis=0)
                
                    ensemble_results['Voting Ensemble'] = {
                        'model': None,  # No sklearn model object
                        'model_name': 'Voting Ensemble',
                        'train_scores': self._calculate_scores(y_train, train_voting_pred, task_type),
                        'test_scores': self._calculate_scores(y_test, test_voting_pred, task_type),
                        'predictions': test_voting_pred,
                        'base_models': list(valid_models.keys()),
                        'training_time': 0,
                        'feature_importance': {'importances': {}}
                    }
                
                else:  # Classification
                    # For classification, handle both hard and soft voting
                    from scipy import stats
                
                    all_predictions = []
                    all_probas = []
                
                    for name, data in valid_models.items():
                        model = data['model']
                        scaler = data.get('scaler')
                    
                        # Apply scaling if needed
                        if scaler is not None:
                            X_test_scaled = scaler.transform(X_test)
                            pred = model.predict(X_test_scaled)
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_test_scaled)
                                all_probas.append(proba)
                        else:
                            pred = model.predict(X_test)
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_test)
                                all_probas.append(proba)
                    
                        all_predictions.append(pred)
                
                    # Hard voting - majority vote
                    all_predictions = np.array(all_predictions)
                    voting_pred = stats.mode(all_predictions, axis=0)[0].flatten()
                
                    # Soft voting if we have probabilities
                    voting_proba = None
                    if all_probas:
                        voting_proba = np.mean(all_probas, axis=0)
                
                    ensemble_results['Voting Ensemble'] = {
                        'model': None,
                        'model_name': 'Voting Ensemble',
                        'test_scores': self._calculate_scores(y_test, voting_pred, task_type, voting_proba),
                        'predictions': voting_pred,
                        'probabilities': voting_proba,
                        'base_models': list(valid_models.keys()),
                        'training_time': 0,
                        'feature_importance': {'importances': {}}
                    }
            
            except Exception as e:
                logger.error(f"Failed to create Voting Ensemble: {str(e)}")
        
        # Stacking ensemble
        if 'Stacking' in methods:
            try:
                # Create list of base estimators with proper names
                base_estimators = []
                for name, data in valid_models.items():
                    model = data['model']
                    # Create a pipeline that includes scaling if needed
                    if data.get('scaler') is not None:
                        from sklearn.pipeline import Pipeline
                        pipeline = Pipeline([
                            ('scaler', data['scaler']),
                            ('model', model)
                        ])
                        base_estimators.append((name.replace(' ', '_'), pipeline))
                    else:
                        base_estimators.append((name.replace(' ', '_'), model))
                
                # Create stacking ensemble with a simple meta-learner
                if task_type == 'regression':
                    meta_learner = Ridge(alpha=1.0)
                    stacking_model = StackingRegressor(
                        estimators=base_estimators,
                        final_estimator=meta_learner,
                        cv=3,
                        n_jobs=self.config.computation.n_jobs
                    )
                else:
                    meta_learner = LogisticRegression(max_iter=1000)
                    stacking_model = StackingClassifier(
                        estimators=base_estimators,
                        final_estimator=meta_learner,
                        cv=3,
                        n_jobs=self.config.computation.n_jobs
                    )
                
                # Train stacking model
                stacking_model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = stacking_model.predict(X_train)
                test_pred = stacking_model.predict(X_test)
                test_proba = None
                if hasattr(stacking_model, 'predict_proba'):
                    test_proba = stacking_model.predict_proba(X_test)
                
                ensemble_results['Stacking Ensemble'] = {
                    'model': stacking_model,
                    'model_name': 'Stacking Ensemble',
                    'train_scores': self._calculate_scores(y_train, train_pred, task_type),
                    'test_scores': self._calculate_scores(y_test, test_pred, task_type, test_proba),
                    'predictions': test_pred,
                    'probabilities': test_proba,
                    'base_models': list(valid_models.keys()),
                    'meta_learner': type(meta_learner).__name__,
                    'training_time': 0,
                    'feature_importance': {'importances': {}}
                }
                
            except Exception as e:
                logger.error(f"Failed to create Stacking Ensemble: {str(e)}")
        
        return ensemble_results
    
    def validate_shap_mathematically(self, shap_values: np.ndarray, model, X_test: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate SHAP values using mathematical properties"""
        details = {
            'additivity_test': False,
            'efficiency_test': False,
            'symmetry_test': False,
            'dummy_test': False,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Test 1: Additivity (most important)
            # sum(SHAP values) + baseline ≈ prediction
            if hasattr(shap_values, 'shape') and len(shap_values.shape) >= 2:
                sample_size = min(10, len(X_test))  # Test on small sample
                sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
                
                predictions = model.predict(X_test.iloc[sample_indices])
                
                if len(shap_values.shape) == 3:  # Multi-class classification
                    # For multi-class, test the predicted class
                    predicted_classes = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions.astype(int)
                    shap_sums = np.array([shap_values[i, :, predicted_classes[i]].sum() for i in range(sample_size)])
                else:  # Regression or binary classification
                    if shap_values.shape[0] > sample_size:
                        shap_subset = shap_values[sample_indices]
                    else:
                        shap_subset = shap_values[:sample_size]
                    shap_sums = shap_subset.sum(axis=1)
                
                # Calculate baseline (expected value)
                try:
                    baseline = np.mean(model.predict(X_test.sample(min(100, len(X_test)))))
                except:
                    baseline = 0
                
                # Check additivity: prediction ≈ baseline + sum(SHAP values)
                expected_predictions = baseline + shap_sums
                actual_predictions = predictions[:len(expected_predictions)]
                
                if len(expected_predictions) == len(actual_predictions):
                    additivity_errors = np.abs(expected_predictions - actual_predictions)
                    mean_error = np.mean(additivity_errors)
                    max_error = np.max(additivity_errors)
                    
                    # More lenient thresholds for complex models
                    if mean_error < 0.1 and max_error < 1.0:
                        details['additivity_test'] = True
                    else:
                        details['issues'].append(f"Additivity test failed: mean_error={mean_error:.6f}, max_error={max_error:.6f}")
                        if mean_error < 1.0:  # Still somewhat reasonable
                            details['warnings'].append("Additivity errors are moderate but acceptable")
                else:
                    details['issues'].append("Could not perform additivity test: dimension mismatch")
            
            # Test 2: Efficiency (baseline should be reasonable)
            try:
                sample_predictions = model.predict(X_test.sample(min(50, len(X_test))))
                prediction_range = np.ptp(sample_predictions)  # peak-to-peak range
                if prediction_range > 0:
                    details['efficiency_test'] = True
                else:
                    details['warnings'].append("Model predictions have no variance")
            except:
                details['warnings'].append("Could not perform efficiency test")
            
            # Test 3: Check for reasonable SHAP value magnitudes
            if hasattr(shap_values, 'shape'):
                max_shap = np.max(np.abs(shap_values))
                mean_shap = np.mean(np.abs(shap_values))
                
                # SHAP values should not be unreasonably large
                if max_shap < 1e6 and mean_shap < 1e3:
                    details['symmetry_test'] = True
                else:
                    details['issues'].append(f"SHAP values have extreme magnitudes: max={max_shap:.2e}, mean={mean_shap:.2e}")
            
            # Overall validation
            tests_passed = sum([details['additivity_test'], details['efficiency_test'], details['symmetry_test']])
            is_valid = tests_passed >= 2 and len(details['issues']) == 0
            
        except Exception as e:
            details['issues'].append(f"Mathematical validation failed: {str(e)}")
            is_valid = False
        
        return is_valid, details
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        model_def = self.model_definitions.get(model_name, {})
        
        info = {
            'name': model_name,
            'category': model_def.get('category', 'Unknown'),
            'type': model_def.get('type', 'Unknown'),
            'supports_multiclass': model_def.get('supports_multiclass', False),
            'feature_importance': model_def.get('feature_importance') is not None,
            'scales_well': model_def.get('scales_well', True),
            'requires_scaling': model_def.get('requires_scaling', False),
            'small_data': model_def.get('small_data', False)
        }
        
        # Add library availability
        if model_name in ['XGBoost']:
            info['available'] = XGBOOST_AVAILABLE
        elif model_name in ['LightGBM']:
            info['available'] = LIGHTGBM_AVAILABLE
        elif model_name in ['CatBoost']:
            info['available'] = CATBOOST_AVAILABLE
        elif model_def.get('category') == 'Bayesian Models':
            info['available'] = BAYESIAN_MODELS_AVAILABLE
        else:
            info['available'] = True
        
        return info
    
    def save_model(self, model_data: Dict, filepath: str) -> bool:
        """Save trained model to disk"""
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model data
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> Dict:
        """Load trained model from disk"""
        try:
            model_data = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return {}
    
    def get_memory_usage_estimate(self) -> float:
        """Get current memory usage estimate in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def cleanup(self):
        """Clean up resources"""
        self.models.clear()
        if hasattr(self, 'interpretability_engine'):
            # Clean up interpretability engine if it has cleanup method
            if hasattr(self.interpretability_engine, 'cleanup'):
                self.interpretability_engine.cleanup()
    
    def get_model_recommendations(self, n_samples: int, n_features: int, 
                                 task_type: str, time_budget: str = 'standard') -> List[str]:
        """Get model recommendations based on data characteristics"""
        recommendations = []
        
        # Small dataset recommendations
        if n_samples < 1000:
            if task_type == 'regression':
                recommendations = ['Ridge', 'Bayesian Ridge', 'ARD Regression', 'Gaussian Process']
            else:
                recommendations = ['Logistic Regression', 'Naive Bayes', 'K-Neighbors']
        
        # Medium dataset recommendations  
        elif n_samples < 10000:
            if task_type == 'regression':
                recommendations = ['Random Forest', 'Ridge', 'Neural Network', 'ARD Regression']
            else:
                recommendations = ['Random Forest', 'Logistic Regression', 'Neural Network']
        
        # Large dataset recommendations
        else:
            if task_type == 'regression':
                recommendations = ['Random Forest', 'XGBoost', 'LightGBM', 'Neural Network']
            else:
                recommendations = ['Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression']
        
        # Filter by availability
        available_models = []
        for model in recommendations:
            if model == 'XGBoost' and not XGBOOST_AVAILABLE:
                continue
            elif model == 'LightGBM' and not LIGHTGBM_AVAILABLE:
                continue
            elif model == 'CatBoost' and not CATBOOST_AVAILABLE:
                continue
            elif model in ['Bayesian Ridge Regression', 'Gaussian Process Regression'] and not BAYESIAN_MODELS_AVAILABLE:
                continue
            available_models.append(model)
        
        return available_models[:6]  # Return top 6 recommendations
    
    def get_training_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate a comprehensive training summary"""
        if not results.get('models'):
            return {'error': 'No models trained'}
        
        # Calculate statistics
        scores = []
        times = []
        for model_data in results['models'].values():
            if 'test_scores' in model_data:
                primary_score = model_data['test_scores'][results['primary_metric']]
                scores.append(primary_score)
            if 'training_time' in model_data:
                times.append(model_data['training_time'])
        
        summary = {
            'total_models': len(results['models']),
            'successful_models': len(scores),
            'failed_models': len(results['models']) - len(scores),
            'best_model': results.get('best_model'),
            'best_score': results.get('best_score'),
            'baseline_score': results.get('baseline_score'),
            'improvement_over_baseline': 0,
            'average_score': np.mean(scores) if scores else 0,
            'score_std': np.std(scores) if scores else 0,
            'total_training_time': sum(times) if times else 0,
            'average_training_time': np.mean(times) if times else 0,
            'task_type': results.get('task_type'),
            'primary_metric': results.get('primary_metric')
        }
        
        # Calculate improvement
        if results.get('best_score') and results.get('baseline_score'):
            baseline = results['baseline_score']
            best = results['best_score']
            if baseline != 0:
                summary['improvement_over_baseline'] = ((best - baseline) / abs(baseline)) * 100
        
        # Add SHAP summary if available
        if results.get('shap_summary'):
            summary['shap_analysis'] = results['shap_summary']
        
        return summary
    
    def export_results_to_csv(self, results: Dict, filepath: str) -> bool:
        """Export training results to CSV"""
        try:
            if not results.get('models'):
                return False
            
            # Prepare data for CSV
            csv_data = []
            for model_name, model_data in results['models'].items():
                row = {
                    'Model': model_name,
                    'Training_Time': model_data.get('training_time', 0),
                    'SHAP_Available': model_data.get('shap_available', False)
                }
                
                # Add test scores
                if 'test_scores' in model_data:
                    for metric, score in model_data['test_scores'].items():
                        row[f'Test_{metric}'] = score
                
                # Add CV scores
                if 'cv_scores' in model_data and model_data['cv_scores']:
                    row['CV_Mean'] = model_data['cv_scores']['mean']
                    row['CV_Std'] = model_data['cv_scores']['std']
                
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False)
            logger.info(f"Results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            return False
    
    def __str__(self) -> str:
        return f"ModelManager(models_available={len(self.model_definitions)}, config={self.config})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Utility functions for model validation and comparison
def compare_model_predictions(model1_data: Dict, model2_data: Dict, 
                            metric: str = 'test_scores') -> Dict[str, Any]:
    """Compare predictions between two models"""
    comparison = {
        'model1': model1_data.get('model_name', 'Unknown'),
        'model2': model2_data.get('model_name', 'Unknown'),
        'better_model': None,
        'score_difference': 0,
        'predictions_correlation': 0
    }
    
    try:
        # Compare scores
        if metric in model1_data and metric in model2_data:
            scores1 = model1_data[metric]
            scores2 = model2_data[metric]
            
            # Use primary metric for comparison
            if isinstance(scores1, dict) and isinstance(scores2, dict):
                primary_metric = list(scores1.keys())[0]  # Use first available metric
                score1 = scores1[primary_metric]
                score2 = scores2[primary_metric]
                
                comparison['score_difference'] = score2 - score1
                comparison['better_model'] = comparison['model2'] if score2 > score1 else comparison['model1']
        
        # Compare predictions
        if 'predictions' in model1_data and 'predictions' in model2_data:
            pred1 = model1_data['predictions']
            pred2 = model2_data['predictions']
            
            if len(pred1) == len(pred2):
                correlation = np.corrcoef(pred1, pred2)[0, 1]
                comparison['predictions_correlation'] = correlation
    
    except Exception as e:
        comparison['error'] = str(e)
    
    return comparison


def get_model_complexity_score(model, model_name: str) -> float:
    """Calculate a complexity score for the model (0-1, higher = more complex)"""
    try:
        # Tree-based models
        if hasattr(model, 'n_estimators'):
            # Ensemble complexity
            base_complexity = min(model.n_estimators / 1000, 0.8)
            if hasattr(model, 'max_depth') and model.max_depth:
                depth_complexity = min(model.max_depth / 20, 0.3)
                return min(base_complexity + depth_complexity, 1.0)
            return base_complexity
        
        # Neural networks
        elif hasattr(model, 'hidden_layer_sizes'):
            if hasattr(model.hidden_layer_sizes, '__len__'):
                n_neurons = sum(model.hidden_layer_sizes)
                n_layers = len(model.hidden_layer_sizes)
            else:
                n_neurons = model.hidden_layer_sizes
                n_layers = 1
            
            neuron_complexity = min(n_neurons / 1000, 0.7)
            layer_complexity = min(n_layers / 10, 0.3)
            return min(neuron_complexity + layer_complexity, 1.0)
        
        # SVM
        elif 'SVM' in model_name:
            if hasattr(model, 'kernel') and model.kernel == 'rbf':
                return 0.7
            elif hasattr(model, 'kernel') and model.kernel == 'poly':
                return 0.8
            else:
                return 0.5
        
        # Linear models
        elif model_name in ['Linear Regression', 'Logistic Regression', 'Ridge', 'Lasso']:
            return 0.2
        
        # Default complexity scores
        complexity_map = {
            'Decision Tree': 0.4,
            'Random Forest': 0.6,
            'Gradient Boosting': 0.7,
            'XGBoost': 0.7,
            'LightGBM': 0.7,
            'CatBoost': 0.7,
            'K-Neighbors': 0.3,
            'Naive Bayes': 0.2,
            'Gaussian Process': 0.8
        }
        
        return complexity_map.get(model_name, 0.5)
    
    except Exception:
        return 0.5  # Default complexity
"""
Feature Availability Checker for AquaVista v6.0
===============================================
Centralized checking for optional dependencies
"""

import logging

logger = logging.getLogger(__name__)

# Check which optional features are available
OPTIONAL_FEATURES = {
    'xgboost': False,
    'lightgbm': False,
    'catboost': False,
    'shap': False,
    'statsmodels': False,
    'ydata_profiling': False,
    'gpu': False,
    'numba': False
}

# Check availability
try:
    import xgboost
    OPTIONAL_FEATURES['xgboost'] = True
    logger.info("✓ XGBoost available")
except ImportError:
    logger.debug("✗ XGBoost not available")

try:
    import lightgbm
    OPTIONAL_FEATURES['lightgbm'] = True
    logger.info("✓ LightGBM available")
except ImportError:
    logger.debug("✗ LightGBM not available")

try:
    import catboost
    OPTIONAL_FEATURES['catboost'] = True
    logger.info("✓ CatBoost available")
except ImportError:
    logger.debug("✗ CatBoost not available")

try:
    import shap
    OPTIONAL_FEATURES['shap'] = True
    logger.info("✓ SHAP available")
except ImportError:
    logger.debug("✗ SHAP not available")

try:
    import statsmodels
    OPTIONAL_FEATURES['statsmodels'] = True
    logger.info("✓ Statsmodels available")
except ImportError:
    logger.debug("✗ Statsmodels not available")

try:
    from ydata_profiling import ProfileReport
    OPTIONAL_FEATURES['ydata_profiling'] = True
    logger.info("✓ YData Profiling available")
except ImportError:
    logger.debug("✗ YData Profiling not available")

try:
    import GPUtil
    if GPUtil.getGPUs():
        OPTIONAL_FEATURES['gpu'] = True
        logger.info("✓ GPU available")
except:
    logger.debug("✗ GPU not available")

try:
    import numba
    OPTIONAL_FEATURES['numba'] = True
    logger.info("✓ Numba acceleration available")
except ImportError:
    logger.debug("✗ Numba not available")


def get_available_features():
    """Get dictionary of available optional features"""
    return OPTIONAL_FEATURES.copy()


def check_feature(feature_name: str) -> bool:
    """Check if a specific feature is available"""
    return OPTIONAL_FEATURES.get(feature_name, False)


def get_missing_features():
    """Get list of missing optional features"""
    return [feat for feat, available in OPTIONAL_FEATURES.items() if not available]


def get_feature_install_command(feature_name: str) -> str:
    """Get pip install command for a missing feature"""
    commands = {
        'xgboost': 'pip install xgboost>=2.0.0',
        'lightgbm': 'pip install lightgbm>=4.0.0',
        'catboost': 'pip install catboost>=1.2.0',
        'shap': 'pip install shap>=0.43.0',
        'statsmodels': 'pip install statsmodels>=0.14.0',
        'ydata_profiling': 'pip install ydata-profiling>=4.5.0',
        'numba': 'pip install numba>=0.58.0'
    }
    return commands.get(feature_name, f'pip install {feature_name}')
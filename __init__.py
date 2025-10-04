"""
AquaVista v6.0 Modules Package
==============================
Machine Learning modules for water quality analysis platform

This package contains all the core modules for the AquaVista platform:
- Configuration management
- Data loading and quality checking
- Data preprocessing and feature engineering
- Model training and management
- Visualization engine
- Report generation
- Model interpretability
- Statistical analysis
- Logging utilities

Author: AquaVista Team
Version: 6.0.0
License: MIT
"""

# Version information
__version__ = '6.0.0'
__author__ = 'AquaVista Team'
__email__ = 'support@aquavista.ai'
__license__ = 'MIT'

# Package metadata
__all__ = [
    # Core modules
    'config',
    'data_loader', 
    'data_quality',
    'data_processor',
    'model_manager',
    'visualization',
    'report_generator',
    'interpretability',
    'statistical_analysis',  # Fixed from 'statistical_analyzer'
    'logging_config',
    'export_manager',  # Added this since it exists
    'performance',     # Added this since it exists
    
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]

# Module descriptions for documentation
MODULE_DESCRIPTIONS = {
    'config': 'Configuration management and settings',
    'data_loader': 'Data loading from various file formats',
    'data_quality': 'Data quality assessment and validation',
    'data_processor': 'Data preprocessing and feature engineering',
    'model_manager': 'Machine learning model training and management',
    'visualization': 'Interactive visualizations using Plotly',
    'report_generator': 'Automated report generation',
    'interpretability': 'Model interpretability and SHAP analysis',
    'statistical_analysis': 'Statistical analysis and hypothesis testing',  # Fixed
    'logging_config': 'Logging configuration and utilities',
    'export_manager': 'Model export and deployment package creation',
    'performance': 'Performance monitoring and resource management'
}

# Package info function
def get_info():
    """Get package information"""
    return {
        'name': 'AquaVista Modules',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'modules': MODULE_DESCRIPTIONS
    }

# Note: We don't import any modules here to avoid circular imports.
# All imports should be done at the point of use in the main application.
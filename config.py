"""
Configuration Module for AquaVista v6.0
======================================
Central configuration management for the platform.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import os
import sys  # ADD THIS LINE
from datetime import datetime


# Custom Exception Classes
class AquaVistaError(Exception):
    """Base exception for AquaVista platform"""
    pass


class DataLoadingError(AquaVistaError):
    """Exception raised for data loading errors"""
    pass


class DataProcessingError(AquaVistaError):
    """Exception raised for data processing errors"""
    pass


class ModelTrainingError(AquaVistaError):
    """Exception raised for model training errors"""
    pass


class VisualizationError(AquaVistaError):
    """Exception raised for visualization errors"""
    pass


class ConfigurationError(AquaVistaError):
    """Exception raised for configuration errors"""
    pass


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    theme: str = "plotly_white"
    color_palette: str = "viridis"
    figure_format: str = "html"
    dpi: int = 300
    interactive: bool = True
    
    # Plot dimensions
    default_height: int = 500
    default_width: int = 800
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["html", "png", "svg"])


@dataclass
class ComputationConfig:
    """Computation and performance settings"""
    n_jobs: int = 8  # Use all available cores
    chunk_size: int = 1000
    memory_limit_mb: int = 4096
    random_state: int = 42
    
    # Timeouts
    training_timeout: int = 3600  # 1 hour
    prediction_timeout: int = 300  # 5 minutes


@dataclass
class DataConfig:
    """Data handling settings"""
    max_rows: Optional[int] = None
    max_columns: Optional[int] = None
    
    # Data quality thresholds
    missing_threshold: float = 0.5  # Drop columns with >50% missing
    correlation_threshold: float = 0.95  # Flag highly correlated features
    
    # Sampling
    enable_sampling: bool = True
    sample_size: int = 100000
    
    # Data types
    infer_dtypes: bool = True
    categorical_threshold: int = 20  # Max unique values for categorical


@dataclass
class ModelConfig:
    """Model training settings"""
    # Cross-validation
    cv_folds: int = 5
    cv_shuffle: bool = True
    
    # Hyperparameter tuning
    tuning_cv: int = 3
    tuning_scoring: str = "auto"
    tuning_n_iter: int = 20
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    
    # Ensemble settings
    ensemble_methods: List[str] = field(default_factory=lambda: ["voting", "stacking"])
    
    # Model selection
    auto_select_models: bool = True
    max_models: int = 10
    
    # Performance thresholds
    min_accuracy: float = 0.6
    min_r2: float = 0.3


@dataclass
class LoggingConfig:
    """Logging settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Path = Path("aquavista_results/logs")
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Console output
    console_output: bool = True
    verbose: bool = False


@dataclass
class ExportConfig:
    """Export and reporting settings"""
    output_dir: Path = Path("aquavista_results")
    include_visualizations: bool = True
    include_code: bool = False
    include_data_sample: bool = True
    save_preprocessors: bool = True
    save_models: bool = True
    model_format: str = "joblib"
    compression: str = "none"  # ADD THIS LINE
    add_timestamp: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"


@dataclass
class Features:
    """Feature flags for optional functionality"""
    # Advanced features
    feature_engineering: bool = True
    auto_feature_selection: bool = True
    shap_analysis: bool = True
    force_shap_display: bool = True  # New: Always show SHAP if calculated
    shap_validation_level: str = "lenient"  # New: strict/lenient/disabled
    lime_analysis: bool = False
    
    # Data features
    auto_data_profiling: bool = True
    outlier_detection: bool = True
    
    # Model features
    auto_ensemble: bool = True
    model_interpretation: bool = True
    uncertainty_quantification: bool = False
    
    # Visualization features
    interactive_plots: bool = True
    three_d_visualizations: bool = False  # Changed from 3d_visualizations
    animated_plots: bool = False
    
    # Performance features
    gpu_acceleration: bool = False
    distributed_training: bool = False
    
    # Analysis features
    statistical_tests: bool = True
    causal_analysis: bool = False
    time_series_analysis: bool = False
    
    # Safety features
    memory_guard: bool = True
    memory_optimization: bool = True  # v6.0: Automatic data type optimization
    smart_caching: bool = True       # v6.0: Intelligent caching system
    handle_imbalance: bool = True    # v6.0: Automatic class imbalance handling
    auto_save: bool = True
    checkpoint_models: bool = True
    
    # UI features
    show_code: bool = False
    show_warnings: bool = True
    show_info: bool = True
    
    # Experimental features
    auto_ml: bool = False
    neural_architecture_search: bool = False
    
    # Other features
    cv_analysis: bool = True


class Config:
    """Main configuration class for AquaVista"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration"""
        # Determine if running as executable or script
        if getattr(sys, 'frozen', False):
            # Running as compiled executable - use folder next to exe
            self.base_dir = Path(sys.executable).parent / "AquaVista_Data"
        else:
            # Running as script - use user home directory
            self.base_dir = Path.home() / ".aquavista"
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Set config path
        self.config_path = config_path or self.base_dir / "config.json"
        
        # Initialize sub-configurations with user directories
        self.visualization = VisualizationConfig()
        self.computation = ComputationConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        
        # Add multicollinearity configuration
        self.multicollinearity = {
            'auto_handle': False,
            'vif_threshold': 10.0,
            'correlation_threshold': 0.9,
            'treatment_method': 'auto'
        }
        
        # Update paths to use user directory
        self.logging = LoggingConfig()
        self.logging.file_path = self.base_dir / "logs"
        
        self.export = ExportConfig()
        self.export.output_dir = self.base_dir / "results"
        
        self.features = Features()
        
        # Performance mode
        self.performance_mode = "balanced"
        
        # Metadata
        self.version = "6.0.0"
        self.created_at = datetime.now()
        
        # Load custom config if exists
        if self.config_path.exists():
            self.load_config()
        
        # Create necessary directories
        self._create_directories()
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.export.output_dir,
            self.export.output_dir / "models",
            self.export.output_dir / "reports",
            self.export.output_dir / "visualizations",
            self.export.output_dir / "data",
            self.logging.file_path,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def update_performance_mode(self, mode: str):
        """Update settings based on performance mode"""
        self.performance_mode = mode
        
        if mode == "memory_efficient":
            self.computation.n_jobs = 1
            self.computation.chunk_size = 500
            self.data.enable_sampling = True
            self.data.sample_size = 50000
            self.model.max_models = 5
            self.features.shap_analysis = False
            self.features.auto_ensemble = False
            
        elif mode == "speed_optimized":
            self.computation.n_jobs = 8
            self.computation.chunk_size = 5000
            self.model.cv_folds = 3
            self.model.tuning_n_iter = 10
            self.features.auto_data_profiling = False
            self.features.statistical_tests = False
            
        elif mode == "accuracy_focused":
            self.model.cv_folds = 10
            self.model.tuning_cv = 5
            self.model.tuning_n_iter = 50
            self.model.ensemble_methods = ["voting", "stacking", "blending"]
            self.features.shap_analysis = True
            self.features.auto_feature_selection = True
            self.features.statistical_tests = True
            
        else:  # balanced
            # Reset to defaults
            self.__init__(self.config_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "performance_mode": self.performance_mode,
            "visualization": self.visualization.__dict__,
            "computation": self.computation.__dict__,
            "data": self.data.__dict__,
            "model": {k: v for k, v in self.model.__dict__.items() if k != 'ensemble_methods'},
            "model_ensemble_methods": self.model.ensemble_methods,
            "multicollinearity": self.multicollinearity,  # Add this line
            "logging": {k: str(v) if isinstance(v, Path) else v for k, v in self.logging.__dict__.items()},
            "export": {k: str(v) if isinstance(v, Path) else v for k, v in self.export.__dict__.items()},
            "features": self.features.__dict__,
            "version": self.version,
            "created_at": self.created_at.isoformat()
        }
    
    def save_config(self, path: Optional[Path] = None):
        """Save configuration to JSON file"""
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def load_config(self, path: Optional[Path] = None):
        """Load configuration from JSON file"""
        load_path = path or self.config_path
        
        if not load_path.exists():
            raise ConfigurationError(f"Config file not found: {load_path}")
        
        with open(load_path, 'r') as f:
            config_dict = json.load(f)
        
        # Update configurations
        self.performance_mode = config_dict.get("performance_mode", "balanced")
        
        # Update sub-configurations
        for key, value in config_dict.get("visualization", {}).items():
            setattr(self.visualization, key, value)
            
        for key, value in config_dict.get("computation", {}).items():
            setattr(self.computation, key, value)
            
        for key, value in config_dict.get("data", {}).items():
            setattr(self.data, key, value)
            
        for key, value in config_dict.get("model", {}).items():
            setattr(self.model, key, value)
            
        if "model_ensemble_methods" in config_dict:
            self.model.ensemble_methods = config_dict["model_ensemble_methods"]
            
        for key, value in config_dict.get("logging", {}).items():
            if key in ["file_path"]:
                setattr(self.logging, key, Path(value))
            else:
                setattr(self.logging, key, value)
                
        for key, value in config_dict.get("export", {}).items():
            if key in ["output_dir"]:
                setattr(self.export, key, Path(value))
            else:
                setattr(self.export, key, value)
                
        for key, value in config_dict.get("features", {}).items():
            setattr(self.features, key, value)
    
    def get_summary(self) -> str:
        """Get configuration summary"""
        return f"""
    AquaVista Configuration Summary
    ==============================
    Version: {self.version}
    Performance Mode: {self.performance_mode}
    Output Directory: {self.export.output_dir}

    Features Enabled:
    - SHAP Analysis: {self.features.shap_analysis}
    - Auto Data Profiling: {self.features.auto_data_profiling}
    - Feature Engineering: {self.features.feature_engineering}
    - Auto Ensemble: {self.features.auto_ensemble}
    - Memory Guard: {self.features.memory_guard}

    Multicollinearity Settings:
    - VIF Threshold: {self.multicollinearity['vif_threshold']}
    - Correlation Threshold: {self.multicollinearity['correlation_threshold']}
    - Auto Handle: {self.multicollinearity['auto_handle']}

    Computation Settings:
    - Parallel Jobs: {self.computation.n_jobs}
    - Random State: {self.computation.random_state}

    Model Settings:
    - CV Folds: {self.model.cv_folds}
    - Max Models: {self.model.max_models}
    """
    
class PortableConfig:
    """Ensure all paths are relative to executable location"""
    
    @staticmethod
    def get_data_dir():
        """Get data directory - works for both dev and compiled"""
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_dir = Path(sys.executable).parent
        else:
            # Running as script
            base_dir = Path.home() / ".aquavista"
        
        # Create data dir in user space (no admin needed)
        data_dir = base_dir / "AquaVista_Data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        return data_dir
    
    @staticmethod
    def get_cache_dir():
        """Get cache directory"""
        cache_dir = PortableConfig.get_data_dir() / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    @staticmethod
    def get_temp_dir():
        """Get temporary directory"""
        temp_dir = PortableConfig.get_data_dir() / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    @staticmethod
    def ensure_user_permissions():
        """Verify we're not running with elevated privileges"""
        if os.name == 'nt':  # Windows
            import ctypes
            if ctypes.windll.shell32.IsUserAnAdmin():
                import warnings
                warnings.warn(
                    "Running with administrator privileges is not recommended. "
                    "AquaVista works best with normal user permissions."
                )
                return False
        return True
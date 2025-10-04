"""
AquaVista v7.0 - Advanced ML Analysis Platform with Bayesian Models
==================================================================
Main Streamlit application file
"""
import os

# Ensure Streamlit uses user-accessible port (>1024)
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'

# Disable telemetry (prevents external connections)
os.environ['STREAMLIT_TELEMETRY'] = 'False'

# Now your regular imports start here...
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import psutil
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
from pathlib import Path
import atexit
import signal
import sys

warnings.filterwarnings("ignore")

# Apply runtime hotfixes
try:
    import hotfixes  # Apply any runtime fixes
    print("Hotfixes applied successfully")
except ImportError:
    pass  # No hotfixes available
except Exception as e:
    print(f"Hotfix error: {e}")

# ---- Arrow-safe helper for Streamlit tables ----
def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make object columns Arrow-friendly for Streamlit display."""
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        out[obj_cols] = out[obj_cols].astype("string")  # pandas nullable string
    return out

# Import custom modules
from modules.config import Config
from modules.data_loader import DataLoader
from modules.data_quality import DataQualityChecker
from modules.data_processor import DataProcessor
from modules.model_manager import ModelManager
from modules.visualization import VisualizationEngine
from modules.report_generator import ReportGenerator
from modules.interpretability import InterpretabilityEngine
from modules.statistical_analysis import StatisticalAnalysis as StatisticalAnalyzer
from modules.logging_config import setup_logging, get_logger
from modules.export_manager import ExportManager
from modules.performance import PerformanceMonitor
from modules.help_texts import HELP_TEXTS
from modules.model_ranking import ModelRankingSystem

# Add Bayesian models import
try:
    from modules.bayesian_models import plot_posterior_predictive_check
    BAYESIAN_MODELS_AVAILABLE = True
except ImportError:
    BAYESIAN_MODELS_AVAILABLE = False
    plot_posterior_predictive_check = None

import atexit
import signal

# Global cleanup handler
def global_cleanup():
    """Global cleanup handler that doesn't rely on session state"""
    try:
        logger.info("Global cleanup initiated")
        # Add any non-session-state cleanup here
    except:
        pass

# Register cleanup handlers
atexit.register(global_cleanup)
# Add the new import for core improvements
# Safe import of core improvements
# Safe import of core improvements module
# Safe import of core improvements module
try:
    from modules.core_improvements import (
        MemoryOptimizer,
        SmartCache,
        UnifiedPipeline,
        handle_class_imbalance,
        TargetEncoder,
    )
    print("[OK] Core improvements module loaded")
except ImportError as e:
    print(f"[WARNING] Core improvements module not available: {e}")
    

    def _norm(label: str) -> str:
        """Normalize label for robust routing (strip emojis/punctuation)."""
        s = "".join(ch for ch in (label or "") if ch.isalnum() or ch.isspace())
        return " ".join(s.split()).lower()

    # Provide proper fallbacks
    MemoryOptimizer = None
    UnifiedPipeline = None
    handle_class_imbalance = None
    TargetEncoder = None
    
    # Create a proper SmartCache fallback that matches the expected interface
    class SmartCacheFallback:
        """Fallback implementation of SmartCache with basic dict functionality"""
        def __init__(self):
            self._cache = {}
            self.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0}
        
        def get(self, key, default=None):
            """Get item with cache statistics tracking"""
            self.cache_stats["total_requests"] += 1
            if key in self._cache:
                self.cache_stats["hits"] += 1
                return self._cache[key]
            else:
                self.cache_stats["misses"] += 1
                return default
        
        def set(self, key, value):
            """Set item in cache"""
            self._cache[key] = value
        
        def __getitem__(self, key):
            return self._cache[key]
        
        def __setitem__(self, key, value):
            self._cache[key] = value
        
        def __contains__(self, key):
            return key in self._cache
        
        def clear(self):
            """Clear cache and reset stats"""
            self._cache.clear()
            self.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0}
    
    SmartCache = SmartCacheFallback
# Safe import wrapper
# Initialize logger
import sys
import logging

TAB_HOME     = "🏠 Home"
TAB_PROCESS  = "📄 Data Processing"
TAB_TRAIN    = "🤖 Model Training"
TAB_RESULTS  = "📊 Results & Analysis"   # pick this and use it everywhere
TAB_PRED     = "🔮 Predictions"
TAB_MANAGE   = "📦 Model Management"     # pick this and use it everywhere
TAB_DOCS     = "📘 Docs"
# Create module-specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid duplicate handlers in Streamlit reruns
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    ch = logging.StreamHandler(stream=sys.stdout)
    try:
        ch.stream.reconfigure(encoding="utf-8")  # Python 3.9+
    except Exception:
        pass  # Fallback silently if reconfigure not supported
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

# Update HELP_TEXTS
HELP_TEXTS.update({
    "bayesian_models": """
    **Bayesian Models** provide probabilistic predictions with uncertainty estimates.
    
    Available models:
    • **Bayesian Linear Regression**: Linear model with uncertainty
    • **Bayesian Ridge**: Automatic relevance determination
    • **Gaussian Process**: Non-parametric Bayesian regression
    • **Bayesian Logistic**: Probabilistic binary classification
    
    Key advantages:
    • Uncertainty quantification
    • Robust to overfitting
    • Principled regularization
    • Feature relevance scores
    """,
    
    "mcmc_settings": """
    **MCMC Settings** control the sampling process:
    
    • **n_samples**: Number of posterior samples (more = better accuracy)
    • **n_chains**: Parallel chains (more = better convergence check)
    • **target_accept**: Target acceptance rate (higher = smaller steps)
    
    Recommendations:
    • Start with defaults (2000 samples, 4 chains)
    • Check R-hat < 1.01 for convergence
    • Increase samples if uncertain
    """
})


class AquaVistaApp:
    """Main application class for AquaVista"""

    def __init__(self):
        try:
            self.fix_bayesian_warnings()
            self.setup_page_config()
            self.initialize_session_state()
            self.load_custom_css()
            
            # Set a flag to track initialization success
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self._initialized = False
            # Don't re-raise - let the app continue with degraded functionality

    def fix_bayesian_warnings(self):
        """Suppress PyTensor compilation warnings"""
        import os
        import warnings
        
        os.environ['PYTHONHASHSEED'] = '0'
        warnings.filterwarnings('ignore', category=UserWarning, module='pytensor')
        warnings.filterwarnings('ignore', message='.*g\\+\\+ not available.*')
        warnings.filterwarnings('ignore', message='.*g\\+\\+ not detected.*')
        
        try:
            import pytensor
            pytensor.config.cxx = ""  # Use Python fallback
        except ImportError:
            pass

    def prevent_memory_leaks(self):
        """Prevent memory leaks by clearing unused session state"""
        try:
            # Clear large objects that might not be needed
            cleanup_keys = []
            
            for key in st.session_state:
                if key.startswith('temp_') or key.endswith('_cache'):
                    cleanup_keys.append(key)
            
            for key in cleanup_keys:
                if key in st.session_state:
                    del st.session_state[key]
                    
            # Force garbage collection periodically
            import gc
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Memory cleanup warning: {e}")

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="AquaVista v7.0",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                "Get Help": "https://github.com/yourusername/aquavista",
                "Report a bug": "https://github.com/yourusername/aquavista/issues",
                "About": "AquaVista v7.0 - Advanced ML Platform with Bayesian Inference",
            },
        )
    def validate_session_state(self, required_keys):
            """
            Validate that required session state keys exist
            
            Args:
                required_keys: List of keys that must exist in session state
                
            Returns:
                tuple: (is_valid, missing_keys)
            """
            missing_keys = []
            for key in required_keys:
                if not hasattr(st.session_state, key) or st.session_state.get(key) is None:
                    missing_keys.append(key)
            
            return len(missing_keys) == 0, missing_keys

    def show_prerequisite_warning(self, step_name, missing_keys):
        """Show a user-friendly warning about missing prerequisites"""
        messages = {
            "data": "📊 Please load your data first in the Home tab",
            "processed_data": "⚙️ Please process your data first in the Data Processing tab",
            "training_results": "🤖 Please train models first in the Model Training tab",
            "quality_report": "✅ Please run data quality check first",
            "statistical_results": "📈 Please generate statistical analysis first"
        }
        
        st.warning(f"### Prerequisites for {step_name}")
        st.write("Please complete the following steps first:")
        
        for key in missing_keys:
            message = messages.get(key, f"Complete {key} step")
            st.write(f"• {message}")
        
        # Add quick navigation buttons
        col1, col2, col3 = st.columns(3)
        
        if "data" in missing_keys:
            with col1:
                if st.button("🏠 Go to Home", type="primary", use_container_width=True):
                    self.current_tab = "🏠 Home"
                    st.rerun()
        
        if "processed_data" in missing_keys:
            with col2:
                if st.button("⚙️ Go to Data Processing", type="primary", use_container_width=True):
                    self.current_tab = "⚙️ Data Processing"
                    st.rerun()
        
        if "training_results" in missing_keys:
            with col3:
                if st.button("🤖 Go to Model Training", type="primary", use_container_width=True):
                    self.current_tab = "🤖 Model Training"
                    st.rerun()
    def initialize_session_state(self):
        """Initialize session state variables with improvements"""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.config = Config()
            st.session_state.data_loader = DataLoader(st.session_state.config)
            st.session_state.quality_checker = DataQualityChecker(
                st.session_state.config
            )
            st.session_state.data_processor = DataProcessor(st.session_state.config)
            st.session_state.model_manager = ModelManager(st.session_state.config)
            st.session_state.viz_engine = VisualizationEngine(st.session_state.config)
            st.session_state.report_generator = ReportGenerator(st.session_state.config)
            st.session_state.interpretability = InterpretabilityEngine(
                st.session_state.config
            )
            st.session_state.statistical_analyzer = StatisticalAnalyzer(
                st.session_state.config
            )
            st.session_state.export_manager = ExportManager(st.session_state.config)
            st.session_state.performance_monitor = PerformanceMonitor(
                st.session_state.config
            )
            st.session_state.current_tab = "🏠 Home"

            # Initialize new features
            st.session_state.cache = SmartCache() if SmartCache != dict else {}
            st.session_state.memory_optimization = True
            st.session_state.handle_imbalance = True

            # Feature engineering state
            # Feature engineering state managed through config only

            logger.info("AquaVista v7.0 initialized successfully with improvements")

            # Feature engineering migration helper (temporary - can be removed after all code is updated)

            # Feature engineering migration helper (temporary - can be removed after all code is updated)

    def load_custom_css(self):
        """Load custom CSS styling"""
        st.markdown(
            """
        <style>
        /* Main styling */
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(120deg, #0066cc, #00a86b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
        }
        /* Metric styling */
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 5px;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 10px rgba(0,0,0,0.2);
        }
        /* Success message */
        .success-message {
            padding: 1rem;
            border-radius: 5px;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        /* Info boxes */
        .info-box {
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 1rem;
            margin: 1rem 0;
        }
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #00a86b;
        }
        /* Popover styling */
        [data-baseweb="popover"] {
            max-width: 500px !important;
        }
    /* Simplified help icon popover styling */
    button[kind="popoverButton"],
    [data-testid="stPopoverButton"] {
        width: 18px !important;
        height: 18px !important;
        min-width: 18px !important;
        min-height: 18px !important;
        padding: 0 !important;
        margin: 0 !important;
        font-size: 10px !important;
        line-height: 1 !important;
        background-color: transparent !important;
        border: 1px solid #cccccc !important;
        border-radius: 50% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 0.4 !important;
        transition: all 0.2s ease !important;
        color: #666666 !important;
        cursor: help !important;
        overflow: hidden !important;
    }

    /* Hover state */
    button[kind="popoverButton"]:hover,
    [data-testid="stPopoverButton"]:hover {
        opacity: 1 !important;
        background-color: #f0f0f0 !important;
        border-color: #666666 !important;
        transform: none !important;
    }

    /* Remove backgrounds from inner elements */
    button[kind="popoverButton"] > div,
    [data-testid="stPopoverButton"] > div {
        background: transparent !important;
    }

    /* Ensure proper sizing in sidebar columns */
    section[data-testid="stSidebar"] .row-widget > div[data-testid="column"]:has(button[kind="popoverButton"]) {
        width: 30px !important;
        max-width: 30px !important;
    }
    
    /* Hover state */
    button[kind="popoverButton"]:hover,
    [data-testid="stPopoverButton"]:hover,
    [data-baseweb="button"][aria-haspopup="true"]:hover,
    div[data-testid="column"] button[aria-haspopup="true"]:hover,
    section[data-testid="stSidebar"] button[aria-haspopup="true"]:hover,
    .stPopover > button:hover {
        opacity: 1 !important;
        background-color: #f0f0f0 !important;
        border-color: #666666 !important;
        transform: none !important;
    }
    
    /* Force remove any Streamlit default styling */
    [data-testid="baseButton-minimal"] {
        background: transparent !important;
        border: 1px solid #cccccc !important;
        width: 18px !important;
        height: 18px !important;
    }
    
    /* Target the specific column structure in sidebar */
    section[data-testid="stSidebar"] .row-widget > div[data-testid="column"]:last-child {
        width: 30px !important;
        max-width: 30px !important;
    }
    
    /* Ensure the popover button container doesn't expand */
    div[data-testid="column"] > div:has(button[kind="popoverButton"]) {
        width: 18px !important;
        height: 18px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    

        </style>
        
        """,
            unsafe_allow_html=True,
        )

    def create_sidebar(self):
        """Create and populate sidebar"""
        with st.sidebar:
            st.title("🌊 AquaVista v7.0")
            st.caption("Advanced ML Platform with Bayesian Inference")

            # Main navigation
            st.header("Navigation")
            self.current_tab = st.radio(
                "Select Module",
                [
                    TAB_HOME,
                    TAB_PROCESS,
                    TAB_TRAIN,
                    TAB_RESULTS,
                    TAB_PRED,
                    TAB_MANAGE,
                    TAB_DOCS,   # new Docs tab
                ],
            )

            st.divider()

            # Configuration section (leave your existing controls here)
            st.header("⚙️ Configuration")
            # ... your config widgets ...

            # Performance mode
            col1, col2 = st.columns([3, 1])
            with col1:
                performance_mode = st.selectbox(
                    "Performance Mode",
                    [
                        "balanced",
                        "memory_efficient",
                        "speed_optimized",
                        "accuracy_focused",
                    ],
                    index=0,
                )
            with col2:
                st.markdown("##")  # Spacing
                with st.popover("ℹ️"):
                    st.markdown(
                        """
                    **Performance Mode Settings**
                    Choose how AquaVista balances resources:
                    • **Balanced**: Optimal trade-off between speed, memory, and accuracy. 
                      Recommended for most use cases.
                    • **Memory Efficient**: Reduces memory usage by limiting parallel 
                      processing and model complexity. Use when system resources are limited.
                    • **Speed Optimized**: Maximizes training speed using all available 
                      cores and simplified models. May use more memory.
                    • **Accuracy Focused**: Enables extensive hyperparameter tuning and 
                      complex models. Slowest but potentially most accurate.
                    """
                    )

            # Update configuration
            if performance_mode != st.session_state.config.performance_mode:
                st.session_state.config.update_performance_mode(performance_mode)
                st.rerun()

            st.divider()

            # Features section
            st.header("⭐ Features")

            # Feature checkboxes with detailed help
            col1, col2 = st.columns([5, 1])
            with col1:
                shap_enabled = st.checkbox(
                    "Enable SHAP Analysis",
                    value=st.session_state.config.features.shap_analysis,
                    key="shap_checkbox",
                )
            with col2:
                with st.popover("ℹ️"):
                    st.markdown(
                        """
                    **SHAP (SHapley Additive exPlanations) Analysis**
                    SHAP provides advanced model interpretability by calculating the 
                    contribution of each feature to individual predictions.
                    
                    **Benefits:**
                    • Understand why models make specific predictions
                    • Identify most influential features for each sample
                    • Detect potential biases in model decisions
                    • Build trust through explainable AI
                    
                    **Note:** SHAP analysis adds ~20-30 seconds to training time 
                    for the top 3 performing models.
                    """
                    )

            col1, col2 = st.columns([5, 1])
            with col1:
                profiling_enabled = st.checkbox(
                    "Auto Data Profiling",
                    value=st.session_state.config.features.auto_data_profiling,
                    key="profiling_checkbox",
                )
            with col2:
                with st.popover("ℹ️"):
                    st.markdown(
                        """
                    **Automatic Data Profiling**
                    Generates comprehensive statistical analysis of your dataset using 
                    the ydata-profiling library.
                    
                    **Includes:**
                    • Descriptive statistics for all features
                    • Missing value analysis and patterns
                    • Correlation matrices and heatmaps
                    • Distribution plots and histograms
                    • Duplicate detection
                    • Data quality warnings
                    
                    **Output:** HTML report saved in results folder
                    **Note:** May take 1-5 minutes for large datasets.
                    """
                    )

            col1, col2 = st.columns([5, 1])
            with col1:
                cv_enabled = st.checkbox(
                    "CV Visualizations",
                    value=st.session_state.config.features.cv_analysis,
                    key="cv_checkbox",
                )
            with col2:
                with st.popover("ℹ️"):
                    st.markdown(
                        """
                    **Cross-Validation Visualizations**
                    Provides detailed analysis of model performance across different 
                    data splits during cross-validation.
                    
                    **Visualizations include:**
                    • Score distribution across CV folds
                    • Model stability analysis (variance)
                    • Learning curves showing training/validation convergence
                    • Fold-wise performance comparison
                    
                    **Benefits:**
                    • Detect overfitting early
                    • Assess model stability
                    • Understand data complexity
                    • Optimize training set size
                    """
                    )

            col1, col2 = st.columns([5, 1])
            with col1:
                memory_guard = st.checkbox(
                    "Memory Guard",
                    value=st.session_state.config.features.memory_guard,
                    key="memory_checkbox",
                )

            with col2:
                with st.popover("ℹ️"):
                    st.markdown(
                        """
                    **Memory Guard Protection**
                    Monitors system resources and prevents out-of-memory errors during 
                    model training and data processing.
                    
                    **Features:**
                    • Real-time memory usage monitoring
                    • Automatic garbage collection
                    • Warning alerts at 75% memory usage
                    • Training pause at 90% to prevent crashes
                    • Smart batch processing for large datasets
                    
                    **Recommended for:**
                    • Large datasets (>100k rows)
                    • Limited RAM systems (<16GB)
                    • Training multiple complex models
                    """
                    )

            # Fixed: Proper feature engineering checkbox handling
            col1, col2 = st.columns([5, 1])
            with col1:
                feature_eng_enabled = st.checkbox(
                    "Feature Engineering",
                    value=st.session_state.config.features.feature_engineering,
                    key="feature_eng_checkbox",
                )
            with col2:
                with st.popover("ℹ️"):
                    st.markdown(
                        """
                    **Automatic Feature Engineering**
                    Creates new features by combining existing ones to potentially 
                    improve model performance.
                    
                    **Transformations applied:**
                    • **Polynomial features**: X², √X for non-linear relationships
                    • **Interaction features**: X₁ × X₂ to capture feature synergies
                    • **Ratio features**: X₁ ÷ X₂ for relative relationships
                    • **Statistical aggregates**: Rolling means, cumulative sums
                    
                    **Benefits:**
                    • Captures complex patterns
                    • Improves model accuracy
                    • Reduces feature engineering effort
                    
                    **Trade-offs:**
                    • Increases feature count (e.g., 10 → 50+ features)
                    • Longer training time
                    • Risk of overfitting with small datasets
                    
                    **Disable if:**
                    • Dataset is very small (<100 samples)
                    • Features are already engineered
                    • Interpretability is critical
                    """
                    )

            # Update configuration with new settings
            if shap_enabled != st.session_state.config.features.shap_analysis:
                st.session_state.config.features.shap_analysis = shap_enabled
            if (
                profiling_enabled
                != st.session_state.config.features.auto_data_profiling
            ):
                st.session_state.config.features.auto_data_profiling = profiling_enabled
            if cv_enabled != st.session_state.config.features.cv_analysis:
                st.session_state.config.features.cv_analysis = cv_enabled
            if memory_guard != st.session_state.config.features.memory_guard:
                st.session_state.config.features.memory_guard = memory_guard
                logger.info(f"[MEMORY GUARD] Memory Guard CHANGED to: {memory_guard}")
            # Fixed: Proper feature engineering state management
            if (
                feature_eng_enabled
                != st.session_state.config.features.feature_engineering
            ):
                st.session_state.config.features.feature_engineering = (
                    feature_eng_enabled
                )

            # Add Bayesian Features section
            st.divider()
            st.header("🎲 Bayesian Features")
            col1, col2 = st.columns([5, 1])
            with col1:
                uncertainty_plots = st.checkbox(
                    "Uncertainty Quantification",
                    value=True,
                    help="Show prediction intervals and uncertainty"
                )
            with col2:
                with st.popover("ℹ️"):
                    st.markdown("""
                    **Uncertainty Quantification**
                    
                    Bayesian models provide uncertainty estimates:
                    • **Prediction intervals**: Range of likely values
                    • **Posterior distributions**: Parameter uncertainty
                    • **Model comparison**: WAIC, LOO metrics
                    
                    Benefits:
                    • Know when predictions are uncertain
                    • Better decision making
                    • Risk assessment
                    """)
            # Update configuration
            st.session_state.config.features.uncertainty_quantification = uncertainty_plots

            # Advanced options
            with st.expander("⚙ Advanced Options", expanded=False):
                st.markdown("### Advanced Configuration")

                # Random seed
                col1, col2 = st.columns([3, 1])
                with col1:
                    random_seed = st.number_input(
                        "Random Seed",
                        min_value=0,
                        max_value=999999,
                        value=st.session_state.config.computation.random_state,
                        help="Set random seed for reproducibility",
                    )
                with col2:
                    st.markdown("##")
                    with st.popover("ℹ️"):
                        st.markdown(
                            """
                        **Random Seed Control**
                        Sets the random number generator seed to ensure reproducible results 
                        across multiple runs.
                        
                        **Affects:**
                        • Train/test data splitting
                        • Cross-validation fold assignment
                        • Random hyperparameter search
                        • Bootstrap sampling in ensemble methods
                        • Neural network weight initialization
                        
                        **Best practices:**
                        • Use same seed for comparing models
                        • Document seed used in experiments
                        • Try multiple seeds to assess stability
                        • Default: 42 (a common choice in ML)
                    """
                        )

                # Parallel jobs
                col1, col2 = st.columns([3, 1])
                with col1:
                    n_jobs = st.slider(
                        "Parallel Jobs",
                        min_value=1,
                        max_value=psutil.cpu_count(),
                        value=(
                            st.session_state.config.computation.n_jobs
                            if st.session_state.config.computation.n_jobs > 0
                            else 8
                        ),
                        help="Number of CPU cores to use",
                    )
                with col2:
                    st.markdown("##")
                    with st.popover("ℹ️"):
                        st.markdown(
                            f"""
                        **Parallel Processing Configuration**
                        Controls how many CPU cores are used for parallel operations.
                        
                        **Your system:** {psutil.cpu_count()} cores available
                        
                        **Guidelines:**
                        • **1 core**: Minimal resource usage, slowest
                        • **2-4 cores**: Good for background processing
                        • **{psutil.cpu_count()//2} cores**: Balanced (recommended)
                        • **{psutil.cpu_count()} cores**: Maximum speed
                        
                        **Affects:**
                        • Model training speed (Random Forest, XGBoost)
                        • Cross-validation
                        • Hyperparameter search
                        • Feature importance calculation
                        
                        **Note:** Using all cores may make system unresponsive
                        """
                        )

                # Logging level
                col1, col2 = st.columns([3, 1])
                with col1:
                    log_level = st.selectbox(
                        "Logging Level",
                        ["DEBUG", "INFO", "WARNING", "ERROR"],
                        index=1,
                        help="Set verbosity of log messages",
                    )
                with col2:
                    st.markdown("##")
                    with st.popover("ℹ️"):
                        st.markdown(
                            """
                        **Logging Verbosity Control**
                        Determines which messages are recorded in the log files.
                        
                        **Levels (from most to least verbose):**
                        • **DEBUG**: All messages including detailed traces
                          - Feature engineering steps
                          - Model parameter details
                          - Memory usage at each step
                          - Use for troubleshooting
                        • **INFO**: General operational messages
                          - Model training progress
                          - Data processing summaries
                          - Performance metrics
                          - Recommended for normal use
                        • **WARNING**: Potential issues
                          - High memory usage
                          - Convergence warnings
                          - Data quality issues
                        • **ERROR**: Only critical failures
                          - Model training failures
                          - Data loading errors
                          - System errors
                        
                        **Log location:** aquavista_results/logs/
                    """
                        )

                # Update advanced settings
                if random_seed != st.session_state.config.computation.random_state:
                    st.session_state.config.computation.random_state = random_seed
                if n_jobs != st.session_state.config.computation.n_jobs:
                    st.session_state.config.computation.n_jobs = n_jobs
                if log_level != st.session_state.config.logging.level:
                    st.session_state.config.logging.level = log_level

            # Advanced features expander
            with st.expander("🚀 Advanced Features", expanded=False):
                st.markdown("### Performance Optimizations")

                col1, col2 = st.columns([5, 1])
                with col1:
                    memory_optimization = st.checkbox(
                        "Memory Optimization",
                        value=st.session_state.get("memory_optimization", True),
                        help="Automatically optimize data types to reduce memory usage",
                    )
                with col2:
                    with st.popover("ℹ️"):
                        st.markdown(
                            """
                        **Memory Optimization**
                        Automatically converts data types to their most efficient forms:
                        • Int64 → Int8/16/32 based on value range
                        • Float64 → Float32 when precision allows
                        • Object → Category for low cardinality strings
                        
                        **Benefits:**
                        • 30-50% memory reduction
                        • Faster processing
                        • Larger datasets support
                    """
                        )

                col1, col2 = st.columns([5, 1])
                with col1:
                    smart_caching = st.checkbox(
                        "Smart Caching",
                        value=True,
                        help="Cache expensive computations for faster reruns",
                    )
                with col2:
                    with st.popover("ℹ️"):
                        st.markdown(
                            """
                        **Smart Caching System**
                        Intelligently caches results of expensive operations:
                        • Feature engineering
                        • Model training
                        • Statistical calculations
                        
                        **Benefits:**
                        • 2-3x speedup on repeated operations
                        • Automatic cache invalidation
                        • Disk and memory caching
                    """
                        )

                col1, col2 = st.columns([5, 1])
                with col1:
                    handle_imbalance = st.checkbox(
                        "Handle Class Imbalance",
                        value=st.session_state.get("handle_imbalance", True),
                        help="Automatically handle imbalanced classification datasets",
                    )
                with col2:
                    with st.popover("ℹ️"):
                        st.markdown(
                            """
                        **Class Imbalance Handling**
                        Automatically detects and handles imbalanced classes:
                        • Calculates imbalance ratio
                        • Applies class weights to models
                        • Supports SMOTE (if installed)
                        
                        **When applied:**
                        • Imbalance ratio > 3:1
                        • Classification tasks only
                        • Improves minority class predictions
                    """
                        )

                # Update configuration
                st.session_state.config.features.memory_optimization = (
                    memory_optimization
                )
                st.session_state.config.features.smart_caching = smart_caching
                st.session_state.memory_optimization = memory_optimization
                st.session_state.handle_imbalance = handle_imbalance

            st.divider()

            # Resource monitor
            st.header("📊 System Resources")

            # Create metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            col1, col2 = st.columns(2)
            with col1:
                memory_color = (
                    "🔴"
                    if memory.percent > 80
                    else "🟡" if memory.percent > 60 else "🟢"
                )
                st.metric(
                    "Memory Usage",
                    f"{memory.percent:.1f}%",
                    f"{memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB",
                    help="Current RAM usage. Red >80%, Yellow >60%",
                )
                st.caption(
                    f"{memory_color} {memory.available / (1024**3):.1f}GB available"
                )

            with col2:
                cpu_color = (
                    "🔴" if cpu_percent > 80 else "🟡" if cpu_percent > 60 else "🟢"
                )
                st.metric(
                    "CPU Usage",
                    f"{cpu_percent:.1f}%",
                    f"{psutil.cpu_count()} cores",
                    help="Current CPU usage across all cores",
                )
                st.caption(f"{cpu_color} {psutil.cpu_freq().current:.0f} MHz")

            # Storage info
            if hasattr(psutil, "disk_usage"):
                disk = psutil.disk_usage("/")
                st.metric(
                    "Disk Space",
                    f"{disk.percent:.1f}% used",
                    f"{disk.free / (1024**3):.1f}GB free",
                    help="Available storage for results and models",
                )

            if st.session_state.config.features.memory_guard:
                # Get system health from performance monitor
                health_status = st.session_state.performance_monitor.get_system_health()

                # Show alerts if any
                recent_alerts = list(st.session_state.performance_monitor.alerts)[-3:]
                if recent_alerts:
                    st.warning("⚠ Recent Alerts:")
                    for alert in recent_alerts:
                        st.caption(f"• {alert['message']}")

                # Session stats
                session_stats = st.session_state.performance_monitor.get_session_stats()
                if session_stats["models_trained"] > 0:
                    st.caption(
                        f"📊 Session: {session_stats['runtime']} | Models: {session_stats['models_trained']}"
                    )

            st.divider()

            # Quick actions
            st.header("⚡ Quick Actions")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Clear Cache",
                    use_container_width=True,
                    help="Clear Streamlit cache to free memory",
                ):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Cache cleared!")
                    st.rerun()

            with col2:
                if st.button(
                    "📁 Open Results",
                    use_container_width=True,
                    help="Open results folder in file explorer",
                ):
                    try:
                        import os
                        import platform

                        results_path = Path("aquavista_results")
                        results_path.mkdir(parents=True, exist_ok=True)

                        if platform.system() == "Windows":
                            os.startfile(results_path)
                        elif platform.system() == "Darwin":  # macOS
                            os.system(f"open {results_path}")
                        else:  # Linux
                            os.system(f"xdg-open {results_path}")
                    except Exception as e:
                        st.error(f"Could not open results folder: {str(e)}")

            # Add cache management display
            if st.session_state.get("cache"):
                cache_stats = st.session_state.cache.cache_stats
                if cache_stats["hits"] > 0 or cache_stats["misses"] > 0:
                    hit_rate = (
                        cache_stats["hits"]
                        / (cache_stats["hits"] + cache_stats["misses"])
                        * 100
                    )
                    st.sidebar.caption(f"🚀 Cache: {hit_rate:.0f}% hit rate")
    def _show_page(self, title: str, fn, *args, **kwargs):
        """Run a page function and show any error in the UI instead of a blank screen."""
        import traceback
        st.subheader(title)
        try:
            fn(*args, **kwargs)
        except Exception as e:
            st.error(f"{title} failed with an error:")
            st.exception(e)      # shows full traceback inline
            # Optional: also echo to the terminal
            traceback.print_exc()

    def run_docs(self):
        import streamlit as st
        st.title("📘 AquaVista Manuals")

        # Set where your docs live (project root or ./docs)
        base = Path(__file__).parent  # project folder
        # If you put them in a 'docs' folder, use: base = Path(__file__).parent / "docs"

        manuals = [
            ("Modeling Manual", base / "AquaVista_Modeling_Manual_v6.md"),
            ("Statistical Analysis Manual", base / "AquaVista_Stat_Analysis_Manual_v6.md"),
        ]

        # Optional downloadable assets if you've saved them
        combined_pdf = (base / "AquaVista_Manuals_v6_combined.pdf")
        bundle_zip   = (base / "AquaVista_Manuals_v6_bundle.zip")

        for title, path in manuals:
            st.markdown(f"### {title}")
            if path.exists():
                md = path.read_text(encoding="utf-8")
                with st.expander(f"View {title}", expanded=False):
                    st.markdown(md)
                st.download_button(
                    f"⬇️ Download {title} (Markdown)",
                    data=md.encode("utf-8"),
                    file_name=path.name,
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.info(f"Could not find {path.name}. Place it in the project folder and reload.")

            st.divider()

        # Optional: offer combined PDF and/or ZIP if present
        cols = st.columns(2)
        if combined_pdf.exists():
            with cols[0]:
                st.download_button(
                    "📄 Download Combined PDF",
                    data=combined_pdf.read_bytes(),
                    file_name=combined_pdf.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        if bundle_zip.exists():
            with cols[1]:
                st.download_button(
                    "🗜️ Download Docs Bundle (ZIP)",
                    data=bundle_zip.read_bytes(),
                    file_name=bundle_zip.name,
                    mime="application/zip",
                    use_container_width=True,
                )


    def run_home_page(self):
        """Display home page"""
        # Header
        st.markdown(
            '<h1 class="main-header">🌊 AquaVista v7.0</h1>', unsafe_allow_html=True
        )
        st.markdown("### Advanced Machine Learning Platform with Bayesian Inference")

        # Welcome message
        st.markdown(
            """
        <div class="info-box">
        Welcome to AquaVista v7.0! This platform provides comprehensive analysis 
        using state-of-the-art machine learning techniques including Bayesian inference. 
        Upload your data to get started.
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Key features
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🎯 Key Features")
            st.markdown(
                """
            - **30+ ML Models** available
            - **Bayesian inference models**
            - **Automated preprocessing**
            - **Feature engineering**
            - **Hyperparameter optimization**
            - **Ensemble methods**
            """
            )

        with col2:
            st.markdown("#### 📊 Analysis Tools")
            st.markdown(
                """
            - **Data quality assessment**
            - **Statistical analysis**
            - **SHAP interpretability**
            - **Uncertainty quantification**
            - **Cross-validation**
            - **Performance metrics**
            """
            )

        with col3:
            st.markdown("#### 📈 Visualizations")
            st.markdown(
                """
            - **Interactive plots**
            - **Feature importance**
            - **Model comparison**
            - **Error analysis**
            - **Posterior distributions**
            - **Custom reports**
            """
            )

        st.divider()

        # Quick start guide
        st.header("🚀 Quick Start Guide")

        # File upload section
        st.subheader("1. Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "parquet", "json"],
            help="Supported formats: CSV, Excel, Parquet, JSON",
        )

        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                try:
                    # Load data based on file type
                    file_extension = uploaded_file.name.split(".")[-1].lower()

                    if file_extension == "csv":
                        df = pd.read_csv(uploaded_file)
                    elif file_extension in ["xlsx", "xls"]:
                        df = pd.read_excel(uploaded_file)
                    elif file_extension == "parquet":
                        df = pd.read_parquet(uploaded_file)
                    elif file_extension == "json":
                        df = pd.read_json(uploaded_file)
                    else:
                        st.error(f"Unsupported file type: {file_extension}")
                        return

                    # Store in session state
                    st.session_state.data = df
                    st.session_state.data_loaded = True

                    # Display success message
                    st.success(f"✓ Data loaded successfully! Shape: {df.shape}")

                    # Quick data preview
                    with st.expander("📋 Data Preview", expanded=True):
                        st.dataframe(arrow_safe(df), use_container_width=True)


                    # Data quality check
                    st.subheader("2. Data Quality Assessment")
                    if st.button("Run Quality Check", type="primary"):
                        with st.spinner("Analyzing data quality..."):
                            quality_report = (
                                st.session_state.quality_checker.check_data_quality(df)
                            )
                            st.session_state.quality_report = quality_report

                            # Display quality metrics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                quality_score = quality_report["overall_quality_score"]
                                st.metric(
                                    "Quality Score",
                                    f"{quality_score:.1f}%",
                                    delta=(
                                        f"{quality_score - 70:.1f}%"
                                        if quality_score >= 70
                                        else None
                                    ),
                                    delta_color=(
                                        "normal" if quality_score >= 70 else "inverse"
                                    ),
                                )

                            with col2:
                                missing_percent = quality_report["missing_percentage"]
                                st.metric(
                                    "Missing Data",
                                    f"{missing_percent:.1f}%",
                                    delta=(
                                        f"{missing_percent:.1f}%"
                                        if missing_percent > 0
                                        else "0%"
                                    ),
                                    delta_color="inverse",
                                )

                            with col3:
                                st.metric(
                                    "Duplicate Rows",
                                    quality_report["duplicate_rows"],
                                    delta=(
                                        None
                                        if quality_report["duplicate_rows"] == 0
                                        else f"+{quality_report['duplicate_rows']}"
                                    ),
                                    delta_color="inverse",
                                )

                            with col4:
                                st.metric(
                                    "High Cardinality",
                                    len(
                                        quality_report.get(
                                            "high_cardinality_features", []
                                        )
                                    ),
                                    help="Features with many unique values",
                                )

                            # Recommendations
                            if quality_report.get("recommendations"):
                                st.info(
                                    "💡 **Recommendations:**\n"
                                    + "\n".join(quality_report["recommendations"])
                                )

                    # Next steps
                    st.subheader("3. Next Steps")
                    st.info(
                        """
                    ⭐ Your data is ready for processing! Navigate to:
                    - **📊 Data Processing** - Configure preprocessing and feature engineering
                    - **🤖 Model Training** - Train and compare ML models
                    - **📈 Results & Analysis** - View detailed results and visualizations
                    """
                    )

                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    logger.error(f"File upload error: {str(e)}")

        # Sample data option
        else:
            st.info("👆 Upload your data to begin, or try our sample dataset below")

            if st.button(
                "🎲 Load Sample Data", help="Load a sample dataset for demonstration"
            ):
                # Create generic sample data
                np.random.seed(42)
                n_samples = 1000

                sample_data = pd.DataFrame(
                    {
                        "Feature_A": np.random.normal(50, 15, n_samples),
                        "Feature_B": np.random.normal(100, 25, n_samples),
                        "Feature_C": np.random.uniform(0, 1, n_samples),
                        "Feature_D": np.random.randint(1, 10, n_samples),
                        "Category_1": np.random.choice(
                            ["Type_A", "Type_B", "Type_C", "Type_D"], n_samples
                        ),
                        "Category_2": np.random.choice(
                            ["Group_1", "Group_2", "Group_3"], n_samples
                        ),
                        "Metric_1": np.random.exponential(2, n_samples),
                        "Metric_2": np.random.gamma(2, 2, n_samples),
                        "Target": np.random.normal(
                            75, 15, n_samples
                        ),  # Target variable
                    }
                )

                st.session_state.data = sample_data
                st.session_state.data_loaded = True
                st.success("✓ Sample data loaded successfully!")
                st.rerun()

    def run_data_processing(self):
        """Run data processing workflow"""
        # Validate prerequisites
        is_valid, missing = self.validate_session_state(["data"])
        if not is_valid:
            self.show_prerequisite_warning("Data Processing", missing)
            return

        st.header("⚙ Data Processing Pipeline")

        # Get current data
        df = st.session_state.data
        
        # Apply memory optimization if enabled and available
        if st.session_state.memory_optimization and MemoryOptimizer is not None:
            with st.spinner("Optimizing memory usage..."):
                try:
                    # Calculate original memory
                    original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
                    
                    # Use the correct method name: optimize_dtypes
                    # Note: MemoryOptimizer.optimize_dtypes is a static method, so we don't need an instance
                    df_optimized = MemoryOptimizer.optimize_dtypes(df, verbose=True)
                    
                    # Calculate new memory
                    new_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2  # MB
                    
                    # Replace the original dataframe
                    df = df_optimized
                    st.session_state.data = df
                    
                    # Show memory savings
                    memory_saved = original_memory - new_memory
                    reduction_pct = (memory_saved / original_memory) * 100 if original_memory > 0 else 0
                    
                    # Only show success message if there was actual memory savings
                    if memory_saved > 0:
                        st.success(f"""
                        ✓ Memory optimization complete!
                        - Original size: {original_memory:.1f} MB
                        - Optimized size: {new_memory:.1f} MB
                        - Memory saved: {memory_saved:.1f} MB ({reduction_pct:.1f}%)
                        """)
                    else:
                        st.info("Data types are already optimized.")
                    
                    logger.info(f"Memory optimization: {original_memory:.1f}MB -> {new_memory:.1f}MB")
                    
                except Exception as e:
                    st.warning(f"Memory optimization failed: {str(e)}. Continuing with original data.")
                    logger.warning(f"Memory optimization error: {str(e)}")
        
        # Processing configuration
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Select Target Variable")
            target_column = st.selectbox(
                "Target Column",
                options=df.columns.tolist(),
                index=len(df.columns) - 1,
                help=HELP_TEXTS.get(
                    "target_column", "Select the column you want to predict"
                ),
            )

        with col2:
            st.subheader("Select Features")
            # Get recommended features
            if hasattr(st.session_state, "quality_report"):
                recommended = st.session_state.quality_report.get(
                    "recommended_features", []
                )
                default_features = [col for col in recommended if col != target_column]
            else:
                default_features = [col for col in df.columns if col != target_column]

            feature_columns = st.multiselect(
                "Feature Columns",
                options=[col for col in df.columns if col != target_column],
                default=default_features,
                help=HELP_TEXTS.get(
                    "feature_columns",
                    "Select columns to use as features for prediction",
                ),
            )

        if not feature_columns:
            st.error("Please select at least one feature column!")
            return

        # Advanced preprocessing options
        with st.expander("Advanced Preprocessing Options", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                missing_strategy = st.selectbox(
                    "Missing Value Strategy",
                    [
                        "auto",
                        "mean",
                        "median",
                        "mode",
                        "forward_fill",
                        "interpolate",
                        "drop",
                    ],
                    help=HELP_TEXTS.get(
                        "missing_strategy", "How to handle missing values"
                    ),
                )

                scaling_method = st.selectbox(
                    "Feature Scaling",
                    [
                        "auto (best for each model)",
                        "standard",
                        "minmax",
                        "robust",
                        "none",
                    ],
                    help="Auto mode tests all scalers (Standard, MinMax, Robust, None) on each model and selects the best based on cross-validation performance. Other options force a specific scaler for all models.",
                )

            with col2:
                encoding_method = st.selectbox(
                    "Categorical Encoding",
                    ["auto", "onehot", "target", "ordinal"],
                    help="How to encode categorical variables",
                )

                remove_outliers = st.checkbox(
                    "Remove Outliers",
                    value=False,
                    help="Remove statistical outliers using IQR method",
                )

            with col3:
                if remove_outliers:
                    outlier_threshold = st.slider(
                        "Outlier Threshold (IQR)",
                        min_value=1.0,
                        max_value=5.0,
                        value=1.5,
                        step=0.5,
                        help="IQR multiplier for outlier detection",
                    )
                else:
                    outlier_threshold = 1.5

                test_size = st.slider(
                    "Test Set Size",
                    min_value=0.1,
                    max_value=0.4,
                    value=0.2,
                    step=0.05,
                    help="Proportion of data for testing",
                )
        
        # ENHANCED: Multicollinearity Analysis with treatment options
        with st.expander("🔗 Multicollinearity Analysis & Treatment", expanded=False):
            # Ensure config section exists BEFORE accessing it
            if not hasattr(st.session_state.config, 'multicollinearity'):
                st.session_state.config.multicollinearity = {
                    'auto_handle': False,
                    'vif_threshold': 10.0,
                    'correlation_threshold': 0.9,
                    'treatment_method': 'auto',
                    'ard_aggressiveness': 10000
                }
            
            col1, col2 = st.columns([3, 1])
            with col1:
                auto_handle_multicollinearity = st.checkbox(
                    "Enable Multicollinearity Detection",
                    value=st.session_state.config.multicollinearity.get('auto_handle', False),
                    help="Automatically detect and handle multicollinear features during data processing"
                )
                st.session_state.config.multicollinearity['auto_handle'] = auto_handle_multicollinearity
                
            with col2:
                with st.popover("ℹ️"):
                    st.markdown("""
                    **Multicollinearity Pipeline**
                    
                    Our 3-phase approach automatically handles correlated features:
                    
                    **Phase 1: Early Detection (Data Processing)**
                    • Calculates VIF and correlation before feature engineering
                    • Removes obvious redundancies (correlation > 0.95)
                    • Limits polynomial features when correlation is high
                    
                    **Phase 2: Smart Feature Engineering**
                    • Reduces polynomial degree when VIF > 20
                    • Skips interaction terms for highly correlated features
                    • Creates conservative features to avoid inflating VIF
                    
                    **Phase 3: Model Selection (Training)**
                    • Prioritizes ARD Regression when VIF > 20
                    • Recommends Ridge/Lasso for moderate multicollinearity
                    • Applies treatment if VIF > 50 (extreme cases)
                    
                    **Why this matters:**
                    • Prevents unstable coefficients in linear models
                    • Improves SHAP interpretability
                    • Reduces overfitting risk
                    • Makes feature importance more reliable
                    """)
            
            if auto_handle_multicollinearity:
                st.markdown("##### Detection Thresholds")
                col1, col2 = st.columns(2)
                with col1:
                    vif_threshold = st.slider(
                        "VIF Threshold", 
                        min_value=1.0, max_value=20.0, 
                        value=st.session_state.config.multicollinearity.get('vif_threshold', 10.0),
                        step=1.0,
                        help="VIF > 10 = multicollinearity, VIF > 20 = severe"
                    )
                    st.session_state.config.multicollinearity['vif_threshold'] = vif_threshold
                    
                with col2:
                    correlation_threshold = st.slider(
                        "Correlation Threshold",
                        min_value=0.1, max_value=0.95, 
                        value=st.session_state.config.multicollinearity.get('correlation_threshold', 0.9),
                        step=0.05,
                        help="Pairs with |correlation| above this are flagged"
                    )
                    st.session_state.config.multicollinearity['correlation_threshold'] = correlation_threshold
                
                st.markdown("##### Treatment Strategy")
                col1, col2 = st.columns([3, 1])
                with col1:
                    treatment_method = st.selectbox(
                        "Treatment Method",
                        options=['auto', 'ard', 'correlation', 'vif', 'none'],
                        index=['auto', 'ard', 'correlation', 'vif', 'none'].index(
                            st.session_state.config.multicollinearity.get('treatment_method', 'auto')
                        ),
                        help="How to handle detected multicollinearity"
                    )
                    st.session_state.config.multicollinearity['treatment_method'] = treatment_method
                
                with col2:
                    with st.popover("ℹ️"):
                        st.markdown("""
                        **Treatment Methods:**
                        
                        • **Auto**: Intelligent selection based on severity
                          - VIF 10-20: Model prioritization only
                          - VIF 20-50: ARD + regularized models
                          - VIF > 50: ARD treatment + feature removal
                        
                        • **ARD**: Automatic Relevance Determination
                          - Uses Bayesian ARD Regression
                          - Automatically zeros out irrelevant features
                          - Best for high multicollinearity (VIF > 20)
                        
                        • **Correlation**: Remove by correlation
                          - Drops one feature from correlated pairs
                          - Fast and interpretable
                          - Good for moderate cases
                        
                        • **VIF**: Remove by VIF score
                          - Iteratively removes highest VIF features
                          - More thorough than correlation
                          - Takes longer to compute
                        
                        • **None**: Detection only
                          - No automatic treatment
                          - Relies on model selection only
                        """)
                
                # Show treatment details based on selection
                if treatment_method in ['auto', 'ard']:
                    st.markdown("##### ARD Settings")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        ard_threshold = st.slider(
                            "ARD Aggressiveness (threshold_lambda)",
                            min_value=1,
                            max_value=100000,
                            value=st.session_state.config.multicollinearity.get('ard_aggressiveness', 10000),
                            step=1000,
                            help="Higher = more aggressive feature pruning"
                        )
                        st.session_state.config.multicollinearity['ard_aggressiveness'] = ard_threshold
                    
                    with col2:
                        with st.popover("ℹ️"):
                            st.markdown("""
                            **ARD Aggressiveness:**
                            
                            Controls how aggressively ARD zeros out features:
                            
                            • **1,000-5,000**: Conservative
                              - Keeps most features
                              - Use when features are valuable
                            
                            • **10,000**: Default (recommended)
                              - Balanced pruning
                              - Good general setting
                            
                            • **50,000-100,000**: Aggressive
                              - Heavy feature pruning
                              - Use for severe multicollinearity
                              - May remove important features
                            
                            ARD automatically determines feature relevance
                            by analyzing precision (inverse variance) of 
                            feature weights. Features with precision below
                            threshold_lambda are considered irrelevant.
                            """)
                    
                    # Show what will happen
                    if treatment_method == 'auto':
                        st.info("""
                        **Auto mode will:**
                        1. Detect multicollinearity during data processing
                        2. If VIF > 20: Prioritize ARD in model selection
                        3. If VIF > 50: Apply ARD treatment before training
                        4. Always recommend regularized models (Ridge/Lasso)
                        """)
                    else:
                        st.info("""
                        **ARD mode will:**
                        1. Apply ARD Regression treatment if VIF > 50
                        2. Use ARD to automatically identify irrelevant features
                        3. Remove features with low precision weights
                        4. Prioritize ARD Regression in model selection
                        """)
                
                elif treatment_method == 'correlation':
                    st.info("""
                    **Correlation-based treatment:**
                    • Finds all feature pairs with |correlation| > threshold
                    • Keeps feature with higher correlation to target
                    • Fast and interpretable approach
                    • Good for moderate multicollinearity
                    """)
                
                elif treatment_method == 'vif':
                    st.info("""
                    **VIF-based treatment:**
                    • Iteratively calculates VIF for all features
                    • Removes feature with highest VIF > threshold
                    • Repeats until all VIF values acceptable
                    • More thorough but slower than correlation
                    """)
                
                else:  # none
                    st.warning("""
                    **Detection only mode:**
                    • Will detect and report multicollinearity
                    • No automatic feature removal
                    • Relies on model selection to handle correlations
                    • Tree-based models handle multicollinearity well
                    • Linear models may have unstable coefficients
                    """)
                    
        # -------------------------------
        # --- Manual Process Data button (always visible) ---
        clicked = st.button(
            "🚀 Process Data",
            key="process_data_manual",
            type="primary",
            use_container_width=True,
        )

        if clicked:
            with st.spinner("Processing data..."):
                try:
                    # --- Build preprocessing config ---
                    preprocess_config = {
                        "missing_strategy": missing_strategy,
                        "scaling_method": scaling_method,
                        "encoding_method": encoding_method,
                        "remove_outliers": remove_outliers,
                        "outlier_threshold": outlier_threshold,
                        "test_size": test_size,
                        "feature_engineering": st.session_state.config.features.feature_engineering,
                        "handle_imbalance": st.session_state.handle_imbalance,
                    }

                    # Multicollinearity options (only if enabled)
                    if auto_handle_multicollinearity:
                        preprocess_config.update({
                            "auto_handle_multicollinearity": True,
                            "vif_threshold": vif_threshold,
                            "correlation_threshold": correlation_threshold,
                            "treatment_method": treatment_method,
                        })
                        if treatment_method in ["auto", "ard"]:
                            preprocess_config["ard_aggressiveness"] = ard_threshold

                    # Optional memory optimization
                    if st.session_state.get("memory_optimization") and (globals().get("MemoryOptimizer") is not None):
                        preprocess_config["optimize_memory"] = True

                    # --- Process data ---
                    processed_data = st.session_state.data_processor.process_data(
                        df=df,
                        target_column=target_column,          # <— keep as target
                        feature_columns=feature_columns,      # <— keep as features
                        preprocess_config=preprocess_config,
                    )

                    # Persist + annotate
                    processed_data["target_column"] = target_column
                    st.session_state.processed_data = processed_data

                    # --- Success summary ---
                    st.success(
                        f"""
        ✓ Data processing complete!
        - Features: {processed_data['X_train'].shape[1]} {'(with engineering)' if st.session_state.config.features.feature_engineering else '(original only)'}
        - Training samples: {processed_data['X_train'].shape[0]}
        - Test samples: {processed_data['X_test'].shape[0]}
        - Task type: {processed_data['task_type']}
        """
                    )

                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    logger.error(f"Data processing error: {str(e)}", exc_info=True)


                    # -----------------------------------------
                    # Feature engineering summary (if applicable)
                    # -----------------------------------------
                    if st.session_state.config.features.feature_engineering:
                        original_features = len(feature_columns)
                        engineered_features = processed_data["X_train"].shape[1]
                        st.info(
                            f"""
        ⚙ **Feature Engineering Applied**
        - Original features: {original_features}
        - After engineering: {engineered_features}
        - New features created: {engineered_features - original_features}
        """
                        )

                    # -------------------
                    # Processing report  (container-safe: no inner expander)
                    # -------------------
                    with st.container():
                        st.subheader("📋 Processing Report")
                        report = processed_data.get("processing_report", {"steps": [], "warnings": []})

                        # Display steps
                        st.markdown("#### Processing Steps")
                        for step in report.get("steps", []):
                            # Show step name + details dict
                            st.markdown(f"- **{step.get('step','(unnamed)')}**: `{step}`")

                        # Display warnings
                        if report.get("warnings"):
                            st.warning("**Warnings:**\n" + "\n".join(report["warnings"]))

                    # -------------------
                    # Feature information (container-safe)
                    # -------------------
                    with st.container():
                        st.subheader("📊 Feature Information")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Numerical Features:**")
                            st.write(processed_data.get("numerical_features", []))
                        with col2:
                            st.markdown("**Categorical Features:**")
                            st.write(processed_data.get("categorical_features", []))

                    # -------------------
                    # Statistical analysis
                    # -------------------
                    if st.session_state.config.features.auto_data_profiling:
                        if st.button("📊 Generate Statistical Report", key="stats_report_btn"):
                            with st.spinner("Generating statistical analysis..."):
                                try:
                                    # Store current state
                                    current_target = target_column

                                    # Create combined dataframe (X_train + y_train)
                                    analysis_df = pd.DataFrame(processed_data["X_train"].copy())
                                    analysis_df[current_target] = processed_data["y_train"]

                                    # Run analysis
                                    stats_results = st.session_state.statistical_analyzer.run_complete_analysis(
                                        data=analysis_df,
                                        target_column=current_target,
                                        task_type=processed_data["task_type"],
                                    )
                                    st.session_state.statistical_results = stats_results

                                    # Display insights in a container (prevents rerun flicker)
                                    with st.container():
                                        st.subheader("🎯 Key Statistical Findings")
                                        for insight in stats_results.get("insights", []):
                                            st.info(insight)

                                    # Display key metrics
                                    col1, col2, col3 = st.columns(3)

                                    # Non-normal features
                                    with col1:
                                        if "distribution_analysis" in stats_results:
                                            non_normal = sum(
                                                1
                                                for _, d in stats_results["distribution_analysis"].items()
                                                if d.get("normality", {}).get("is_normal") is False
                                            )
                                            st.metric(
                                                "Non-normal features",
                                                non_normal,
                                                help="Features that don't follow normal distribution",
                                            )

                                    # Highly correlated pairs
                                    with col2:
                                        if "correlation_analysis" in stats_results:
                                            high_corr = len(
                                                stats_results["correlation_analysis"].get("high_correlations", [])
                                            )
                                            st.metric(
                                                "Highly correlated pairs",
                                                high_corr,
                                                help="Feature pairs with correlation > 0.8",
                                            )

                                    # Features with outliers
                                    with col3:
                                        if "outlier_analysis" in stats_results:
                                            outlier_features = len(
                                                stats_results["outlier_analysis"]
                                                .get("summary", {})
                                                .get("features_with_outliers", [])
                                            )
                                            st.metric(
                                                "Features with outliers",
                                                outlier_features,
                                                help="Features containing statistical outliers",
                                            )

                                except Exception as e:
                                    st.error(f"Error generating statistical report: {str(e)}")
                                    logger.error(f"Statistical analysis error: {str(e)}", exc_info=True)

                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    logger.error(f"Data processing error: {str(e)}", exc_info=True)


    def run_model_training(self):
        """Run model training workflow"""
        is_valid, missing = self.validate_session_state(["processed_data"])
        if not is_valid:
            self.show_prerequisite_warning("Model Training", missing)
            return

        st.header("🤖 Model Training")

        # Get processed data info
        processed_data = st.session_state.processed_data
        n_features = processed_data["X_train"].shape[1]
        n_samples = processed_data["X_train"].shape[0]
        task_type = processed_data["task_type"]

        # Display data info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Training Samples", n_samples)
        with col2:
            st.metric("Features", n_features)
        with col3:
            st.metric("Task Type", task_type.capitalize())
        with col4:
            if task_type == "classification":
                st.metric("Classes", processed_data["n_classes"])

        st.divider()

        # Model selection
        st.subheader("🎯 Model Selection")

        # Get available models
        available_models = st.session_state.model_manager.get_available_models(
            task_type, n_samples
        )

        # Model presets
        col1, col2 = st.columns([2, 1])
        with col1:
            preset = st.selectbox(
                "Quick Selection Presets",
                [
                    "🎯 Recommended",  # <-- THIS IS THE PROBLEMATIC LINE
                    "⚡ Fast Models", 
                    "🏆 High Accuracy",
                    "🔬 All Models",
                    "Custom Selection",
                ],
                help="Choose a preset or select custom models",
            )

        with col2:
            base_models = sum(len(models) for models in available_models.values())
            st.metric("Available Models", base_models)

        # Get selected models based on preset
        if preset != "Custom Selection":
            # DEBUG: Let's see what's happening
            st.write(f"DEBUG: preset = '{preset}'")
            st.write(f"DEBUG: task_type = '{task_type}'")  
            st.write(f"DEBUG: available_models keys = {list(available_models.keys())}")
            st.write(f"DEBUG: available_models = {available_models}")

            selected_models = st.session_state.model_manager.get_model_preset(
                preset, available_models, task_type
            )

            st.write(f"DEBUG: selected_models returned = {selected_models}")
            st.write(f"DEBUG: number of selected_models = {len(selected_models)}")
            
            # FIXED: Show count even if 0
            model_count = len(selected_models) if selected_models else 0
            st.info(f"Selected {model_count} models from {preset} preset")
            
            # NEW: Add explanatory text box for why these models were recommended
            if model_count > 0:
                explanation = self.get_preset_explanation(preset, task_type, n_samples, n_features, selected_models)
                with st.expander("ℹ️ Why these models were recommended", expanded=False):
                    st.markdown(explanation)
            else:
                st.warning("No models selected from preset. This may indicate a configuration issue.")
                
        else:
            # Custom model selection
            selected_models = []

            # Display models by category with explanations
            for category, models in available_models.items():
                with st.expander(f"{category} Models ({len(models)})", expanded=False):
                    # Add category explanation
                    category_help = self.get_category_explanation(category, task_type)
                    if category_help:
                        st.info(category_help)
                    
                    cols = st.columns(3)
                    for idx, model in enumerate(models):
                        with cols[idx % 3]:
                            if st.checkbox(model, key=f"model_{model}"):
                                selected_models.append(model)

        if not selected_models:
            st.warning("Please select at least one model to train!")
            return

        # Add info when Bayesian models are selected
        if any("Bayesian" in model for model in selected_models):
            st.info("""
            🎲 **Bayesian Models Selected**: These models will provide uncertainty estimates 
            along with predictions. Training may take longer due to MCMC sampling.
            """)

        # Training configuration
        st.subheader("⚙️ Training Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            use_cv = st.checkbox(
                "Cross-Validation",
                value=True,
                help="Use k-fold cross-validation for robust evaluation",
            )
            if use_cv:
                cv_folds = st.slider(
                    "CV Folds",
                    min_value=2,
                    max_value=10,
                    value=5,
                    help=HELP_TEXTS.get("cv_folds", "Number of cross-validation folds"),
                )
            else:
                cv_folds = None

        with col2:
            optimize_hyperparams = st.checkbox(
                "Hyperparameter Tuning",
                value=True,
                help="Automatically optimize model hyperparameters",
            )
            if optimize_hyperparams:
                tuning_budget = st.selectbox(
                    "Tuning Budget",
                    ["quick", "standard", "extensive"],
                    index=1,
                    help="Time/resource budget for hyperparameter search",
                )
            else:
                tuning_budget = None

        with col3:
            ensemble_methods = st.multiselect(
                "Ensemble Methods",
                ["Voting", "Averaging", "Stacking"],
                default=["Voting"],
                help="Create ensemble models from base models",
            )

        # Show total models to be trained (including ensemble)
        if ensemble_methods:
            ensemble_count = len(
                [
                    m
                    for m in ensemble_methods
                    if m in ["Voting", "Averaging", "Stacking"]
                ]
            )
            if "Averaging" in ensemble_methods and task_type != "regression":
                ensemble_count -= 1
            total_models = len(selected_models) + ensemble_count
            st.info(
                f"🤖 Training {len(selected_models)} base models + {ensemble_count} ensemble models = {total_models} total"
            )
        else:
            # Get scaling method from processed data
            scaling_method = processed_data.get("processing_report", {}).get(
                "config", {}
            ).get("scaling_method") or processed_data.get("preprocess_config", {}).get(
                "scaling_method", "auto"
            )
            if scaling_method == "auto (best for each model)":
                scaler_tests = len(selected_models) * 4  # 4 scalers per model
                st.info(
                    f"🤖 Training {len(selected_models)} models × 4 scalers = {scaler_tests} configurations"
                )
                st.caption(
                    "Each model will be tested with Standard, MinMax, Robust, and No scaling"
                )
            else:
                st.info(f"🤖 Training {len(selected_models)} models")

        # Memory estimation
        estimated_memory = st.session_state.model_manager.estimate_memory_usage(
            selected_models, n_samples, n_features
        )

        # Check memory before training
        memory_ok = True
        if st.session_state.config.features.memory_guard:
            available_memory = psutil.virtual_memory().available / (1024**2)
            if estimated_memory > available_memory * 0.8:
                st.warning(
                    f"""
                ⚠️ Memory Warning: Estimated usage ({estimated_memory:.0f}MB) may exceed 
                available memory ({available_memory:.0f}MB). Consider selecting fewer models or enabling Memory Guard.
                """
                )
                memory_ok = False

        # Train models button
        if st.button(
            "🚀 Train Models",
            type="primary",
            use_container_width=True,
            disabled=not memory_ok,
        ):
            # Log training start
            st.session_state.performance_monitor.log_event(
                "training_start",
                {
                    "models": selected_models,
                    "n_models": len(selected_models),
                    "n_samples": n_samples,
                    "n_features": n_features,
                },
            )

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)

            with st.spinner("Training models..."):
                try:
                    start_time = time.time()

                    # Train models
                    training_results = st.session_state.model_manager.train_models(
                        selected_models=selected_models,
                        processed_data=processed_data,
                        use_cv=use_cv,
                        cv_folds=cv_folds,
                        optimize_hyperparams=optimize_hyperparams,
                        tuning_budget=tuning_budget,
                        ensemble_methods=ensemble_methods,
                        progress_callback=update_progress,
                    )

                    # Store results
                    st.session_state.training_results = training_results

                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()

                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time

                    # Log training complete
                    st.session_state.performance_monitor.log_event(
                        "training_complete",
                        {
                            "duration": elapsed_time,
                            "best_model": training_results["best_model"],
                            "best_score": training_results["best_score"],
                        },
                    )

                    # Display success message
                    st.success(
                        f"""
                    ✓ Training complete in {elapsed_time:.1f} seconds!
                    - Models trained: {len(training_results['models'])}
                    - Best model: {training_results['best_model']}
                    - Best score: {training_results['best_score']:.4f}
                    """
                    )

                    # Display model performance summary
                    st.subheader("📊 Model Performance Summary")

                    summary_df = st.session_state.viz_engine.create_performance_summary(
                        training_results
                    )

                    # Display with styling
                    display_df = arrow_safe(summary_df)
                    st.dataframe(
                        display_df.style.format(precision=4).highlight_max(
                            subset=[col for col in display_df.columns if "Test" in col],
                            color="lightgreen",
                        ),
                        use_container_width=True,
                    )

                    # Quick visualizations
                    col1, col2 = st.columns(2)

                    with col1:
                        st.plotly_chart(
                            st.session_state.viz_engine.create_performance_chart(
                                training_results
                            ),
                            use_container_width=True,
                        )

                    with col2:
                        # Model Performance Heatmap
                        if 'training_results' in st.session_state and st.session_state.training_results:
                            heatmap_fig = st.session_state.viz_engine.create_model_comparison_heatmap(
                                st.session_state.training_results
                            )
                            if heatmap_fig:
                                st.plotly_chart(heatmap_fig, use_container_width=True, key="model_comparison_heatmap")
                        else:
                            st.info("Train models first to see the performance heatmap")

                    # Model insights
                    # Model Ranking Table
                    st.subheader("📊 Model Performance Ranking")

                    # Create ranking table
                    primary_metric = training_results.get("primary_metric", "r2")

                    # Determine if lower is better
                    lower_is_better_metrics = [
                        "mae",
                        "mse",
                        "rmse",
                        "neg_mean_absolute_error",
                        "neg_mean_squared_error",
                        "neg_root_mean_squared_error",
                    ]
                    is_lower_better = any(
                        metric in primary_metric.lower()
                        for metric in lower_is_better_metrics
                    )

                    # Collect model data for ranking
                    ranking_data = []
                    for model_name, model_data in training_results["models"].items():
                        test_scores = model_data.get("test_scores", {})
                        primary_score = test_scores.get(
                            primary_metric,
                            float("inf") if is_lower_better else float("-inf"),
                        )

                        # Handle sklearn negative scores
                        if primary_metric.startswith("neg_") and primary_score < 0:
                            primary_score = -primary_score

                        # Skip invalid scores
                        if not (float("-inf") < primary_score < float("inf")):
                            continue

                        # Get all metrics for display
                        metrics_dict = {"Model": model_name}
                        for metric, value in test_scores.items():
                            # Clean up metric names and values
                            if metric.startswith("neg_") and value < 0:
                                value = -value

                            # Format metric name
                            display_metric = (
                                metric.replace("neg_mean_", "")
                                .replace("_", " ")
                                .upper()
                            )
                            if display_metric == "SQUARED ERROR":
                                display_metric = "MSE"
                            elif display_metric == "ABSOLUTE ERROR":
                                display_metric = "MAE"
                            elif display_metric == "ROOT MEAN SQUARED ERROR":
                                display_metric = "RMSE"

                            metrics_dict[display_metric] = value

                        # Add training time
                        metrics_dict["Time (s)"] = model_data.get("training_time", 0)

                        ranking_data.append((primary_score, metrics_dict))

                    # Sort by primary metric
                    ranking_data.sort(key=lambda x: x[0], reverse=not is_lower_better)

                    # Create dataframe for display
                    if ranking_data:
                        import pandas as pd

                        # Add rank column
                        display_data = []
                        for rank, (score, data) in enumerate(ranking_data, 1):
                            data["Rank"] = str(rank)
                            # Add medal emoji for top 3
                            if rank == 1:
                                data["Rank"] = str(rank) + " 🥇"
                            elif rank == 2:
                                data["Rank"] = str(rank) + " 🥈"
                            elif rank == 3:
                                data["Rank"] = str(rank) + " 🥉"
                            display_data.append(data)

                        # Create DataFrame
                        ranking_df = pd.DataFrame(display_data)

                        # Reorder columns to put Rank and Model first
                        cols = ["Rank", "Model"] + [
                            col
                            for col in ranking_df.columns
                            if col not in ["Rank", "Model"]
                        ]
                        ranking_df = ranking_df[cols]

                        # Display the table with formatting
                        display_rank = arrow_safe(ranking_df)

                        # --- Robust metric column detection (handles names like RMSE_test, r2_score, valid_accuracy) ---
                        loss_keys = ("mae", "mse", "rmse", "mape", "smape")               # lower is better
                        gain_keys = ("r^2", "r2", "r²", "r2_score", "accuracy", "precision", "recall", "f1", "f1_score", "auc")

                        # normalize once (handle r^2 / r² / r^2 variants etc.)
                        cols_norm = {
                            c: (c.lower()
                                .replace("r^2", "r2")
                                .replace("r²", "r2"))
                            for c in display_rank.columns
                        }

                        # pick columns by substring match
                        loss_cols = [c for c, lc in cols_norm.items() if any(k in lc for k in loss_keys)]
                        gain_cols = [c for c, lc in cols_norm.items() if any(k in lc for k in gain_keys)]

                        # drop clearly non-metrics if they slipped in
                        skip = {"rank", "model", "training time (s)"}
                        loss_cols = [c for c in loss_cols if c.lower() not in skip]
                        gain_cols = [c for c in gain_cols if c.lower() not in skip]

                        # if any column landed in both buckets, prefer "gain"
                        overlap = set(loss_cols) & set(gain_cols)
                        if overlap:
                            loss_cols = [c for c in loss_cols if c not in overlap]

                        # keep only numeric columns (styler min/max needs numeric)
                        import pandas as pd
                        loss_cols = [c for c in loss_cols if pd.api.types.is_numeric_dtype(display_rank[c])]
                        gain_cols = [c for c in gain_cols if pd.api.types.is_numeric_dtype(display_rank[c])]

                        styled = (
                            display_rank.style.format(precision=4)
                                .highlight_min(subset=loss_cols or [], color="lightgreen")
                                .highlight_max(subset=gain_cols or [], color="lightgreen")
                        )

                        st.dataframe(
                            styled,
                            use_container_width=True,
                            height=min(600, 35 + len(display_rank) * 35),
                        )

                        # Add performance insights
                        st.markdown("### 🎯 Performance Insights")

                        # Best model
                        best_model = (
                            display_data[0]["Model"] if display_data else "Unknown"
                        )
                        st.success(
                            f"**Best Model**: {best_model} achieves the best {primary_metric.upper()} score"
                        )

                        # Performance spread
                        if len(ranking_data) > 1:
                            best_score = ranking_data[0][0]
                            worst_score = ranking_data[-1][0]

                            if is_lower_better:
                                improvement = (
                                    (worst_score - best_score) / worst_score
                                ) * 100
                                st.info(
                                    f"**Performance Range**: Best model is {improvement:.1f}% better than worst"
                                )
                            else:
                                improvement = (
                                    ((best_score - worst_score) / worst_score) * 100
                                    if worst_score != 0
                                    else 0
                                )
                                st.info(
                                    f"**Performance Range**: {improvement:.1f}% improvement from worst to best model"
                                )

                        # Model categories
                        if len(ranking_data) >= 5:
                            top_5 = [data[1]["Model"] for data in ranking_data[:5]]
                            tree_models = [
                                m
                                for m in top_5
                                if any(
                                    t in m.lower()
                                    for t in [
                                        "tree",
                                        "forest",
                                        "boost",
                                        "xgb",
                                        "catboost",
                                        "lightgbm",
                                    ]
                                )
                            ]
                            ensemble_models = [
                                m
                                for m in top_5
                                if any(
                                    e in m.lower()
                                    for e in [
                                        "ensemble",
                                        "voting",
                                        "stacking",
                                        "average",
                                    ]
                                )
                            ]

                            if tree_models:
                                st.info(
                                    f"🌲 **Tree-based models** in top 5: {', '.join(tree_models)}"
                                )
                            if ensemble_models:
                                st.info(
                                    f"🎯 **Ensemble models** in top 5: {', '.join(ensemble_models)}"
                                )

                    # Save results option
                    if st.button("💾 Save Training Results"):
                        results_path = Path("aquavista_results")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Ensure directory exists
                        model_dir = results_path / "models"
                        model_dir.mkdir(parents=True, exist_ok=True)

                        # Save model
                        model_path = model_dir / f"best_model_{timestamp}.joblib"
                        st.session_state.model_manager.save_model(
                            training_results["best_model_data"], model_path
                        )

                        st.success(f"Results saved to {results_path}")

                except Exception as e:
                    # Clear UI elements FIRST to prevent memory leaks
                    progress_bar.empty()
                    status_text.empty()
                    
                    # THEN show error messages
                    st.error(f"Error during training: {str(e)}")
                    logger.error(f"Model training error: {str(e)}", exc_info=True)

                finally:
                    # Ensure cleanup happens even if an unexpected error occurs
                    if 'progress_bar' in locals():
                        progress_bar.empty()
                    if 'status_text' in locals():
                        status_text.empty()

    def create_scaler_usage_chart(self, training_results):
        """Create a chart showing which scaler was used for each model"""
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np

        # Extract data with explicit type conversion
        model_names = []
        r2_values = []
        scaler_types = []
        colors = []

        # Define scaler groups and colors
        scaler_groups = {
            "None": "No Scaling",
            "StandardScaler": "Standardization",
            "MinMaxScaler": "Min-Max Normalization",
            "RobustScaler": "Robust Scaling",
        }

        color_map = {
            "None": "#2E86AB",  # Blue
            "StandardScaler": "#A23B72",  # Purple
            "MinMaxScaler": "#F18F01",  # Orange
            "RobustScaler": "#C73E1D",  # Red
        }

        # Extract data from training results
        for model_name, model_data in training_results["models"].items():
            if "ensemble" not in model_name.lower():
                # Get R2 score - handle different formats
                r2_score = model_data.get("test_scores", {}).get("r2_score", 0)
                
                # Convert to regular Python float
                if isinstance(r2_score, np.ndarray):
                    r2_score = float(r2_score.item())
                elif hasattr(r2_score, 'item'):
                    r2_score = float(r2_score.item())
                else:
                    r2_score = float(r2_score)
                
                # Get scaler type
                if model_data.get("scaler_type"):
                    scaler_type = model_data["scaler_type"]
                elif model_data.get("scaler") is not None:
                    scaler_type = type(model_data["scaler"]).__name__
                else:
                    scaler_type = "None"
                
                # Append to lists
                model_names.append(model_name)
                r2_values.append(r2_score)
                scaler_types.append(scaler_type)
                colors.append(color_map.get(scaler_type, "#999999"))
        
        # Create DataFrame and sort by R2 score
        df = pd.DataFrame({
            'Model': model_names,
            'R2': r2_values,
            'Scaler': scaler_types,
            'Color': colors
        })
        
        # Sort by R2 score (ascending for horizontal bar chart)
        df = df.sort_values('R2', ascending=True).reset_index(drop=True)
        
        # Create figure
        fig = go.Figure()
        
        # Add a single bar trace with all data
        fig.add_trace(
            go.Bar(
                name='',
                y=df['Model'].tolist(),
                x=df['R2'].tolist(),
                orientation='h',
                marker=dict(
                    color=df['Color'].tolist(),
                    line=dict(width=0.5, color='white')
                ),
                text=[f'{val:.4f}' for val in df['R2']],
                textposition='inside',
                textfont=dict(size=10, color='white'),
                insidetextanchor='end',
                hovertemplate='<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>',
                showlegend=False
            )
        )
        
        # Add legend entries
        legend_added = set()
        for scaler in ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]:
            if scaler in df['Scaler'].values and scaler not in legend_added:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=color_map[scaler],
                            symbol='square'
                        ),
                        showlegend=True,
                        name=scaler_groups[scaler]
                    )
                )
                legend_added.add(scaler)
        
        # Update layout with explicit settings
        fig.update_layout(
            title="Model Performance by Scaler Type Used",
            xaxis=dict(
                title="R² Score",
                range=[0, 1.05],
                constrain='domain',
                constraintoward='left',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                showline=True,
                linewidth=1,
                linecolor='black',
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                tickformat='.1f',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black'
            ),
            yaxis=dict(
                title="Model",
                type='category',
                categoryorder='array',
                categoryarray=df['Model'].tolist(),
                showgrid=False,
                showline=True,
                linewidth=1,
                linecolor='black'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                title=dict(text="Scaler Type", font=dict(size=12)),
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            margin=dict(l=150, r=200, t=50, b=50),
            height=max(500, len(df) * 25),
            bargap=0.2,
            font=dict(size=11)
        )
        
        # Force update to ensure proper rendering
        fig.update_xaxes(range=[0, 1.05])
        fig.update_traces(
            cliponaxis=False,
            selector=dict(type='bar')
        )
        
        # Debug output
        print(f"\nDEBUG - Bar chart data:")
        for i in range(min(5, len(df))):
            print(f"  {df.iloc[i]['Model']}: R² = {df.iloc[i]['R2']:.4f}")
        print(f"  ... (showing first 5 of {len(df)} models)")
        
        return fig
    
    def create_scaler_insights(self, training_results):
        """Create insights about scaler usage"""
        import pandas as pd

        # Analyze scaler usage
        scaler_stats = {
            "None": [],
            "StandardScaler": [],
            "MinMaxScaler": [],
            "RobustScaler": [],
        }
        category_scaler = {}

        for model_name, model_data in training_results["models"].items():
            if "ensemble" not in model_name.lower():
                r2 = model_data["test_scores"].get("r2_score", 0)

                # Get scaler
                # Check for scaler_type first (v6.0), then fall back to scaler object
                if model_data.get("scaler_type"):
                    scaler_type = model_data["scaler_type"]
                elif model_data.get("scaler") is not None:
                    scaler_type = type(model_data["scaler"]).__name__
                else:
                    scaler_type = "None"

                if scaler_type in scaler_stats:
                    scaler_stats[scaler_type].append(r2)

                # Track by category
                model_def = st.session_state.model_manager.model_definitions.get(
                    model_name, {}
                )
                category = model_def.get("category", "Other")
                if category not in category_scaler:
                    category_scaler[category] = {}
                if scaler_type not in category_scaler[category]:
                    category_scaler[category][scaler_type] = []
                category_scaler[category][scaler_type].append(r2)

        # Calculate average R2 by scaler
        scaler_avg = {}
        for scaler, scores in scaler_stats.items():
            if scores:
                scaler_avg[scaler] = sum(scores) / len(scores)

        return scaler_stats, scaler_avg, category_scaler

    def run_results_analysis(self):
        """Display results and analysis"""
        # Validate prerequisites
        is_valid, missing = self.validate_session_state(["training_results", "processed_data"])
        if not is_valid:
            self.show_prerequisite_warning("Results & Analysis", missing)
            return
        
        st.header("📊 Results & Analysis")

        training_results = st.session_state.training_results

        # Create tabs for different analyses
        tabs = st.tabs([
            "📊 Overview",
            "🔍 Model Comparison",
            "🔬 Feature Analysis",
            "📉 Error Analysis",
            "🎨 Custom Visualizations",
            "🎲 Bayesian Analysis",  # New tab
            "🔗 Multicollinearity"  # New tab
        ])

        with tabs[0]:  # Overview
            st.subheader("Training Overview")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Best Model", training_results["best_model"])
            with col2:
                st.metric("Best Score", f"{training_results['best_score']:.4f}")
            with col3:
                st.metric("Total Time", f"{training_results['total_time']:.1f}s")
            with col4:
                st.metric("Models Trained", len(training_results["models"]))

            # Model Ranking Table
            st.subheader("📊 Model Performance Ranking")

            # Create ranking data
            ranking_data = []
            primary_metric = training_results.get("primary_metric", "r2_score")

            for model_name, model_data in training_results["models"].items():
                test_scores = model_data.get("test_scores", {})

                # Create row data
                row = {
                    "Model": model_name,
                    "Training Time (s)": round(model_data.get("training_time", 0), 2),
                }

                # Add all test scores
                for metric, value in test_scores.items():
                    # Clean up metric names
                    display_metric = metric.replace("_", " ").title()
                    if "r2" in metric.lower():
                        display_metric = "R²"
                    elif metric == "mae":
                        display_metric = "MAE"
                    elif metric == "mse":
                        display_metric = "MSE"
                    elif metric == "rmse":
                        display_metric = "RMSE"

                    row[display_metric] = round(value, 4)

                ranking_data.append(row)

            # Create DataFrame and sort by primary metric
            if ranking_data:
                import pandas as pd

                ranking_df = pd.DataFrame(ranking_data)

                # Determine sort order
                primary_col = (
                    "R²" if "R²" in ranking_df.columns else list(ranking_df.columns)[1]
                )
                ascending = primary_col in [
                    "MAE",
                    "MSE",
                    "RMSE",
                ]  # Lower is better for these

                ranking_df = ranking_df.sort_values(primary_col, ascending=ascending)
                ranking_df.reset_index(drop=True, inplace=True)

                # Add rank column with medals
                ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))
                ranking_df["Rank"] = ranking_df["Rank"].apply(
                    lambda x: (
                        f"{x} 🥇"
                        if x == 1
                        else f"{x} 🥈" if x == 2 else f"{x} 🥉" if x == 3 else str(x)
                    )
                )

                # Display the table with formatting
                display_rank2 = arrow_safe(ranking_df)
                st.dataframe(
                    display_rank2.style.format(
                        precision=4,
                        subset=[col for col in display_rank2.columns
                                if col not in ["Rank", "Model", "Training Time (s)"]],
                    ),
                    use_container_width=True,
                    height=min(600, 50 + len(display_rank2) * 35),
                )

                # Add insights
                st.markdown("##### 🎯 Key Insights")
                best_model = ranking_df.iloc[0]["Model"]
                st.success(
                    f"**Best Model:** {best_model} achieves the best {primary_col} score"
                )

                # Training time insight
                fastest_model = ranking_df.loc[
                    ranking_df["Training Time (s)"].idxmin(), "Model"
                ]
                fastest_time = ranking_df["Training Time (s)"].min()
                st.info(
                    f"**Fastest Model:** {fastest_model} trained in just {fastest_time:.1f} seconds"
                )

            # Performance chart
            st.plotly_chart(
                st.session_state.viz_engine.create_performance_chart(training_results),
                use_container_width=True,
            )

            # Detailed summary table
            st.subheader("Detailed Performance Metrics")
            summary_df = st.session_state.viz_engine.create_performance_summary(
                training_results
            )
            st.dataframe(arrow_safe(summary_df), use_container_width=True)


        with tabs[1]:  # Model Comparison
            st.subheader("Model Comparison")

            # Add this code to your aquavista_main.py file
            # In the run_results_analysis method, right after the Model Comparison header
            # Around line 1700, after st.subheader("Model Comparison")

            # Comparison heatmap
            self.create_graph_header_with_help(
                "Model Performance Heatmap (Column-wise Normalized)",
                """
                **Model Performance Comparison Heatmap**
                
                This heatmap visualizes how different models perform across various metrics:
                
                **How to read:**
                • Each row represents a model
                • Each column represents a performance metric
                • Colors indicate performance levels:
                  - 🟦 **Blue/Cool colors**: Lower performance
                  - 🟨 **Yellow/Warm colors**: Medium performance  
                  - 🟥 **Red/Hot colors**: Higher performance
                
                **Metrics shown:**
                • **R² Score**: Coefficient of determination (0-1, higher is better)
                • **MAE**: Mean Absolute Error (lower is better)
                • **RMSE**: Root Mean Square Error (lower is better)
                • **MAPE**: Mean Absolute Percentage Error (lower is better)
                
                **Row normalization**: Each row is normalized to show relative performance
                across metrics, making it easier to compare models with different scales.
                
                **Use this to:**
                • Quickly identify best performers (more red cells)
                • Spot models that excel at specific metrics
                • Find balanced models (consistent colors across metrics)
                """,
            )
            st.plotly_chart(
                st.session_state.viz_engine.create_model_comparison_heatmap(
                    training_results
                ),
                use_container_width=True,
            )

            # Pairwise comparison
            st.subheader("Pairwise Model Comparison")

            model_names = list(training_results["models"].keys())
            col1, col2 = st.columns(2)

            with col1:
                model1 = st.selectbox("First Model", model_names, index=0)
            with col2:
                model2 = st.selectbox(
                    "Second Model", model_names, index=1 if len(model_names) > 1 else 0
                )

            if model1 != model2:
                st.plotly_chart(
                    st.session_state.viz_engine.create_pairwise_comparison(
                        training_results, model1, model2
                    ),
                    use_container_width=True,
                )

            # Cross-validation analysis
            if st.session_state.config.features.cv_analysis:
                st.subheader("Cross-Validation Analysis")

                col1, col2 = st.columns(2)
                with col1:
                    self.create_graph_header_with_help(
                        "Cross-Validation Scores by Model",
                        """
                        **Cross-Validation Performance Distribution**
                        
                        This box plot shows how each model performs across different CV folds:
                        
                        **Understanding the plot:**
                        • **Box**: Middle 50% of scores (25th-75th percentile)
                        • **Line in box**: Median score
                        • **Whiskers**: Range of scores excluding outliers
                        • **Dots**: Individual fold scores or outliers
                        
                        **What it reveals:**
                        • **Model stability**: Smaller boxes = more consistent
                        • **Best typical performance**: Higher median lines
                        • **Risk assessment**: Long whiskers = high variance
                        
                        **Red flags:**
                        • Very wide boxes (unstable performance)
                        • Many outlier dots (inconsistent model)
                        • Large gap between whiskers (unpredictable)
                        
                        **Ideal pattern:**
                        • Small box at high position
                        • Short whiskers
                        • Few or no outliers
                        """,
                    )
                    st.plotly_chart(
                        st.session_state.viz_engine.create_cv_scores_plot(
                            training_results
                        ),
                        use_container_width=True,
                    )
                with col2:
                    self.create_graph_header_with_help(
                        "Model Stability Analysis (CV Fold Variance)",
                        """
                        **Model Stability Ranking**
                        
                        This chart ranks models by their consistency across CV folds:
                        
                        **Metrics shown:**
                        • **CV Standard Deviation**: Variation in scores across folds
                        • **Lower is better**: Less variation = more reliable
                        
                        **Interpretation:**
                        • 🟢 **Low variance (<0.02)**: Very stable model
                        • 🟡 **Medium variance (0.02-0.05)**: Acceptable stability
                        • 🔴 **High variance (>0.05)**: Unstable, risky model
                        
                        **Why stability matters:**
                        • Stable models generalize better to new data
                        • Unstable models may have learned noise
                        • High variance suggests overfitting risk
                        
                        **Use this to:**
                        • Eliminate unreliable models
                        • Choose between similar performers
                        • Assess deployment risk
                        """,
                    )
                    st.plotly_chart(
                        st.session_state.viz_engine.create_stability_analysis(
                            training_results
                        ),
                        use_container_width=True,
                    )

            # Scaler Usage Analysis
            if "models" in training_results and training_results["models"]:
                st.markdown("### ⚙ Scaler Usage and Performance")

                col1, col2 = st.columns([2, 1])

                with col1:
                    self.create_graph_header_with_help(
                        "Model Performance by Scaler Type Used",
                        """
                    **Scaler Impact on Model Performance**
                    
                    This horizontal bar chart shows which scaling method was used for each 
                    model and the resulting R² score:
                    
                    **Scaler types:**
                    • **No Scaling**: Raw features (tree-based models)
                    • **Standardization**: Mean=0, SD=1 (linear models)
                    • **Min-Max**: Scale to [0,1] (distance-based models)
                    • **Robust**: Median/IQR scaling (outlier-resistant)
                    
                    **Bar colors** represent different scaler types
                    **Bar length** shows R² score (longer = better)
                    
                    **Key insights:**
                    • Tree models (RF, XGBoost) typically use no scaling
                    • Linear models benefit from standardization
                    • Neural networks often prefer min-max scaling
                    • Models sensitive to outliers use robust scaling
                    
                    **What to notice:**
                    • Which scaler works best for each model type
                    • Whether scaling improves or hurts performance
                    • Patterns in model categories (tree vs. linear vs. neural)
                    """,
                    )
                scaler_chart = self.create_scaler_usage_chart(training_results)
                st.plotly_chart(scaler_chart, use_container_width=True)

                with col2:
                    # Scaler insights
                    scaler_stats, scaler_avg, category_scaler = (
                        self.create_scaler_insights(training_results)
                    )

                    st.markdown("#### 📊 Scaler Performance")
                    for scaler, avg_r2 in sorted(
                        scaler_avg.items(), key=lambda x: x[1], reverse=True
                    ):
                        scaler_name = {
                            "None": "No Scaling",
                            "StandardScaler": "Standard",
                            "MinMaxScaler": "Min-Max",
                            "RobustScaler": "Robust",
                        }.get(scaler, scaler)

                        count = len(scaler_stats[scaler])
                        st.metric(
                            label=scaler_name,
                            value=f"{avg_r2:.4f}",
                            delta=f"{count} models",
                        )

                # Scaler insights removed - using automatic selection for all models
                # ============= COMPREHENSIVE MODEL RANKING SECTION =============
                if training_results and 'models' in training_results:
                    st.header("🏆 Comprehensive Model Ranking System")
                    
                    # Use case priority selection
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        use_case_priority = st.selectbox(
                            "Select your priority focus:",
                            options=['balanced', 'performance', 'interpretability', 'efficiency'],
                            help="Choose what matters most for your use case"
                        )
                    
                    with col2:
                        if st.button("Generate Rankings", type="primary"):
                            st.session_state['generate_rankings'] = True
                    
                    # Generate rankings
                    if st.session_state.get('generate_rankings', False):
                        with st.spinner("Calculating comprehensive model rankings..."):
                            try:
                                # Initialize ranking system
                                ranking_system = ModelRankingSystem()
                                
                                # Calculate rankings
                                ranking_results = ranking_system.rank_models(
                                    training_results, 
                                    use_case_priority=use_case_priority
                                )
                                
                                if 'error' not in ranking_results:
                                    # Store in session state for persistence
                                    st.session_state['ranking_results'] = ranking_results
                                    st.success(f"Model rankings calculated with {use_case_priority} priority!")
                                else:
                                    st.error(f"Ranking calculation failed: {ranking_results['error']}")
                            
                            except Exception as e:
                                st.error(f"Error calculating rankings: {str(e)}")
                                st.session_state['ranking_results'] = None

                    # Display ranking results
                    if 'ranking_results' in st.session_state and st.session_state['ranking_results']:
                        ranking_results = st.session_state['ranking_results']
                        
                        # Methodology explanation
                        with st.expander("📋 Ranking Methodology & Criteria"):
                            methodology = ranking_results['methodology']
                            st.write("**Scoring Criteria and Weights:**")
                            
                            weights_df = pd.DataFrame([
                                {
                                    'Criterion': criterion.title(),
                                    'Weight': f"{weight:.1%}",
                                    'Description': methodology['scoring_details'][criterion]
                                }
                                for criterion, weight in methodology['criteria_weights'].items()
                            ])
                            st.dataframe(weights_df, hide_index=True, use_container_width=True)
                            
                            st.info(f"**Current Priority:** {ranking_results['use_case_priority'].title()} - "
                                f"Weights adjusted to emphasize your selected focus area.")

                        # Main Rankings Table
                        st.subheader("📊 Model Rankings")
                        
                        # Create rankings dataframe for display
                        rankings_df = pd.DataFrame(ranking_results['rankings'])
                        
                        # Format for better display
                        display_df = rankings_df[['rank', 'model', 'overall_score', 'performance_score', 
                                                'stability_score', 'interpretability_score', 'efficiency_score',
                                                'robustness_score', 'summary']].copy()
                        
                        display_df.columns = ['Rank', 'Model', 'Overall', 'Performance', 'Stability', 
                                            'Interpretability', 'Efficiency', 'Robustness', 'Summary']
                        
                        # Style the dataframe
                        styled_rankings = display_df.style.format({
                            'Overall': '{:.3f}',
                            'Performance': '{:.3f}',
                            'Stability': '{:.3f}', 
                            'Interpretability': '{:.3f}',
                            'Efficiency': '{:.3f}',
                            'Robustness': '{:.3f}'
                        }).background_gradient(
                            subset=['Overall', 'Performance', 'Stability', 'Interpretability', 'Efficiency', 'Robustness'], 
                            cmap='RdYlGn'
                        ).highlight_max(
                            subset=['Overall'], 
                            color='gold'
                        )
                        
                        st.dataframe(styled_rankings, hide_index=True, use_container_width=True)

                        # Quick insights cards
                        st.subheader("🎯 Key Insights")
                        top_model = ranking_results['rankings'][0]
                        
                        insight_col1, insight_col2, insight_col3 = st.columns(3)
                        
                        with insight_col1:
                            st.metric(
                                "🥇 Top Ranked Model", 
                                top_model['model'],
                                f"Score: {top_model['overall_score']:.3f}"
                            )
                        
                        with insight_col2:
                            # Find best performer
                            best_performer = max(ranking_results['rankings'], key=lambda x: x['performance_score'])
                            st.metric(
                                "⚡ Best Performance",
                                best_performer['model'],
                                f"Score: {best_performer['performance_score']:.3f}"
                            )
                        
                        with insight_col3:
                            # Find most interpretable
                            most_interpretable = max(ranking_results['rankings'], key=lambda x: x['interpretability_score'])
                            st.metric(
                                "🔍 Most Interpretable",
                                most_interpretable['model'],
                                f"Score: {most_interpretable['interpretability_score']:.3f}"
                            )

                        # Detailed Model Analysis
                        st.subheader("🔍 Detailed Model Analysis")
                        
                        selected_model = st.selectbox(
                            "Select a model for detailed analysis:",
                            options=[model['model'] for model in ranking_results['rankings']],
                            key="detailed_analysis_selector"
                        )
                        
                        if selected_model:
                            model_info = next(
                                (model for model in ranking_results['rankings'] if model['model'] == selected_model),
                                None
                            )
                            
                            if model_info:
                                detail_col1, detail_col2 = st.columns(2)
                                
                                with detail_col1:
                                    st.write("**✅ Key Strengths:**")
                                    for strength in model_info['key_strengths']:
                                        st.write(f"• {strength}")
                                    
                                    # Score breakdown chart
                                    score_data = {
                                        'Criterion': ['Performance', 'Stability', 'Interpretability', 'Efficiency', 'Robustness'],
                                        'Score': [
                                            model_info['performance_score'],
                                            model_info['stability_score'], 
                                            model_info['interpretability_score'],
                                            model_info['efficiency_score'],
                                            model_info['robustness_score']
                                        ]
                                    }
                                    
                                    score_df = pd.DataFrame(score_data)
                                    fig_radar = px.line_polar(score_df, r='Score', theta='Criterion', 
                                                            line_close=True, title=f'{selected_model} Score Breakdown')
                                    fig_radar.update_traces(fill='toself', fillcolor='rgba(0,100,255,0.1)')
                                    st.plotly_chart(fig_radar, use_container_width=True)
                                
                                with detail_col2:
                                    if model_info['main_weakness']:
                                        st.write("**⚠️ Main Weakness:**")
                                        st.write(f"• {model_info['main_weakness']}")
                                    
                                    if model_info['top_recommendation']:
                                        st.write("**💡 Recommendation:**")
                                        st.write(f"• {model_info['top_recommendation']}")
                                    
                                    # Comparison with top model
                                    if selected_model != top_model['model']:
                                        st.write("**📈 Comparison with Top Model:**")
                                        score_diff = model_info['overall_score'] - top_model['overall_score']
                                        if score_diff < -0.1:
                                            st.write(f"🔻 {abs(score_diff):.3f} points behind top model")
                                        elif score_diff < 0:
                                            st.write(f"🔽 {abs(score_diff):.3f} points behind top model (close competition)")
                                        
                        # Overall Recommendations
                        st.subheader("💡 Recommendations")
                        for i, recommendation in enumerate(ranking_results['recommendations'], 1):
                            st.write(f"{i}. {recommendation}")

                        # Export options
                        st.subheader("📤 Export Rankings")
                        export_col1, export_col2 = st.columns(2)
                        
                        with export_col1:
                            if st.button("📊 Add to Report", key="add_ranking_to_report"):
                                if 'report_sections' not in st.session_state:
                                    st.session_state['report_sections'] = {}
                                st.session_state['report_sections']['model_ranking'] = ranking_results
                                st.success("Rankings added to report sections!")
                        
                        with export_col2:
                            # Create downloadable ranking report
                            ranking_system = ModelRankingSystem()
                            ranking_system.ranking_results = ranking_results
                            report_text = ranking_system.export_ranking_report()
                            
                            st.download_button(
                                label="📄 Download Ranking Report",
                                data=report_text,
                                file_name=f"model_ranking_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )

                    else:
                        st.info("👆 Click 'Generate Rankings' to analyze your trained models with the comprehensive ranking system.")
                        
                        # Show preview of what rankings provide
                        with st.expander("🔍 What does the ranking system evaluate?"):
                            st.write("""
                            The comprehensive ranking system evaluates models across 5 key criteria:
                            
                            **🎯 Performance (35%)**: Test score improvement over baseline
                            **📊 Stability (20%)**: Cross-validation consistency  
                            **🔍 Interpretability (20%)**: SHAP availability and feature importance reliability
                            **⚡ Efficiency (15%)**: Training time and resource usage
                            **🛡️ Robustness (10%)**: Generalization ability (overfitting detection)
                            
                            *Weights adjust based on your selected priority focus.*
                            """)

                # End of integration code for aquavista_main.py
        with tabs[2]:  # Feature Analysis

            # --- Native Feature Importance (from the model itself) ---
            st.subheader("Feature Importance Analysis")

            # Aggregated feature importance across models (if available)
            try:
                agg_fig = st.session_state.viz_engine.create_aggregated_feature_importance(training_results)
                st.plotly_chart(agg_fig, use_container_width=True)
            except Exception:
                pass  # ok if not available

            st.subheader("Feature Importance by Model")

            models_with_fi = []
            for mname, mdata in training_results["models"].items():
                fi = (mdata or {}).get("feature_importance", {})
                imps = (fi or {}).get("importances")
                if imps is not None and len(imps) > 0:
                    models_with_fi.append(mname)

            if models_with_fi:
                selected_model_fi = st.selectbox(
                    "Select model (native feature importance)",
                    models_with_fi,
                    key="fi_model_select",  # avoid clash with SHAP selectbox
                )
                fi_data = training_results["models"][selected_model_fi]["feature_importance"]
                fi_fig = st.session_state.viz_engine.create_feature_importance_plot(fi_data, selected_model_fi)
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                st.info("No models expose native feature importance for this run. Use SHAP below.")

            # --- SHAP Feature Importance ---
            if st.session_state.config.features.shap_analysis:
                st.subheader("SHAP Feature Importance")

                # Summary counters
                shap_summary = training_results.get('shap_summary', {})
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Models Tested", shap_summary.get('total_models', 0))
                with c2: st.metric("SHAP Available", shap_summary.get('successful_calculations', 0))
                with c3: st.metric("SHAP Failed", shap_summary.get('failed_calculations', 0))

                # Build model lists - FIXED: Check shap_available instead of validation
                import numpy as np
                models_with_shap = []
                models_without_shap = []
                model_shap_info = {}

                for model_name, model_data in training_results["models"].items():
                    if "ensemble" in model_name.lower():
                        continue

                    # Check shap_available flag OR usable SHAP values
                    shap_available = model_data.get("shap_available", False)
                    shap_vals = model_data.get("shap_values")
                    
                    has_usable_shap = (
                        shap_available or 
                        (isinstance(shap_vals, np.ndarray) and 
                        shap_vals.size > 0 and 
                        np.abs(shap_vals).max() > 1e-15)
                    )
                    
                    if has_usable_shap and shap_vals is not None:
                        r2 = model_data.get("test_scores", {}).get("r2_score", 0.0)
                        is_validated = model_data.get("shap_validated", False)
                        
                        status_icon = "✓" if is_validated else "⚠"
                        label = f"{status_icon} {model_name} (R² = {r2:.4f})"
                        models_with_shap.append(label)
                        model_shap_info[label] = {
                            "model_name": model_name,
                            "r2_score": r2,
                            "is_validated": is_validated,
                            "shap_values": shap_vals,
                            "max_abs_shap": float(np.abs(shap_vals).max())
                        }
                    else:
                        error_msg = (model_data.get("shap_diagnostics", {}).get("error") or 
                                    model_data.get("shap_error") or "No SHAP calculation")
                        models_without_shap.append({"name": model_name, "reason": error_msg})

                # Main SHAP UI
                if models_with_shap:
                    models_with_shap.sort(key=lambda x: model_shap_info[x]["r2_score"], reverse=True)
                    st.info("✓ = Validated SHAP • ⚠ = Unvalidated but shown • Models sorted by R²")

                    selected_display = st.selectbox("Select Model for SHAP Analysis", models_with_shap)
                    info = model_shap_info[selected_display]
                    shap_values = info["shap_values"]

                    # SHAP quality metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        status = "✅ Valid" if info["is_validated"] else "⚠️ Unvalidated"
                        st.metric("Status", status)
                    with col2:
                        st.metric("Max |SHAP|", f"{info['max_abs_shap']:.2e}")
                    with col3:
                        non_zero_pct = (shap_values != 0).sum() / shap_values.size * 100
                        st.metric("Non-zero", f"{non_zero_pct:.1f}%")
                    with col4:
                        st.metric("Shape", f"{shap_values.shape[0]}×{shap_values.shape[1]}")

                    # Warning for unvalidated
                    if not info["is_validated"]:
                        st.warning("⚠️ Unvalidated SHAP: Use with caution. Consider permutation importance as alternative.")

                    # SHAP plots
                    fig = st.session_state.interpretability.create_shap_summary_plot(
                        shap_values, st.session_state.processed_data["feature_names"]
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Dependence plot
                    st.subheader("SHAP Feature Dependence")
                    feature_for_dependence = st.selectbox(
                        "Select feature for dependence plot",
                        st.session_state.processed_data["feature_names"],
                    )
                    dep_fig = st.session_state.interpretability.create_feature_dependence_plot(
                        shap_values, st.session_state.processed_data["X_test"], feature_for_dependence
                    )
                    st.plotly_chart(dep_fig, use_container_width=True)

                    # Waterfall plot
                    if st.checkbox("Show SHAP Waterfall Plot"):
                        idx = st.number_input(
                            "Select test sample index",
                            min_value=0,
                            max_value=len(st.session_state.processed_data["X_test"]) - 1,
                            value=0,
                        )
                        wf = st.session_state.interpretability.create_shap_waterfall(
                            shap_values[int(idx)],
                            st.session_state.processed_data["feature_names"],
                            st.session_state.processed_data["X_test"].iloc[int(idx)],
                        )
                        st.plotly_chart(wf, use_container_width=True)

                # Models without SHAP
                if models_without_shap:
                    with st.expander(f"Models without SHAP ({len(models_without_shap)})", expanded=False):
                        for item in models_without_shap:
                            st.write(f"**{item['name']}**: {item['reason']}")
                        st.info("Use 'Feature Importance by Model' or permutation importance instead.")
                elif not models_with_shap:
                    st.warning("No SHAP values available. Use built-in feature importance or permutation importance.")
                    
            else:
                st.info("Enable **SHAP Analysis** in the sidebar to view SHAP visualizations.")

            # ADD THE SHAP DEBUGGING CODE HERE:
            st.divider()
            with st.expander("🔧 SHAP Debugging Tools", expanded=False):
                st.markdown("### Debug SHAP for Specific Model")
                st.caption("Use this tool to troubleshoot SHAP calculation issues for specific models")
                
                # Check if we have training results
                if not hasattr(st.session_state, "training_results") or not st.session_state.training_results:
                    st.warning("No trained models available for debugging")
                else:
                    # Model selection for debugging
                    available_models = list(st.session_state.training_results["models"].keys())
                    debug_model = st.selectbox(
                        "Select model to debug",
                        available_models,
                        key="debug_shap_model",
                        help="Choose a model that's having SHAP issues"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        debug_level = st.selectbox(
                            "Debug Level",
                            ["Basic", "Detailed", "Full"],
                            help="Basic: Quick check, Detailed: More info, Full: Everything"
                        )
                    with col2:
                        force_recalc = st.checkbox(
                            "Force Recalculation",
                            help="Ignore cached SHAP values and recalculate"
                        )
                    
                    if st.button("🔍 Debug SHAP for Selected Model", type="primary"):
                        try:
                            with st.spinner("Running SHAP debugging..."):
                                # Your debugging code
                                model_name = debug_model
                                model_data = st.session_state.training_results["models"][model_name]
                                model = model_data["model"]
                                X_test = st.session_state.processed_data["X_test"]
                                y_test = st.session_state.processed_data["y_test"]
                                task_type = st.session_state.processed_data["task_type"]
                                
                                # Display basic info
                                st.success(f"🎯 Debugging Model: **{model_name}**")
                                
                                info_col1, info_col2, info_col3 = st.columns(3)
                                with info_col1:
                                    st.metric("Model Type", type(model).__name__)
                                with info_col2:
                                    st.metric("Task Type", task_type.title())
                                with info_col3:
                                    st.metric("Test Samples", X_test.shape[0])
                                
                                # Run debugging
                                interpretability_engine = st.session_state.interpretability
                                
                                if debug_level in ["Detailed", "Full"]:
                                    st.write("**🔍 Model and Data Analysis:**")
                                    # Debug model and data
                                    interpretability_engine.debug_model_and_data(model, X_test, y_test)
                                
                                # Calculate SHAP with debugging
                                st.write("**⚙️ SHAP Calculation:**")
                                
                                # Clear cached values if requested
                                if force_recalc and "shap_values" in model_data:
                                    del model_data["shap_values"]
                                    st.info("Cleared cached SHAP values")
                                
                                shap_values = interpretability_engine.calculate_shap_values(model, X_test, task_type)
                                
                                # Results
                                if shap_values is not None:
                                    st.success("✅ SHAP values calculated successfully!")
                                    
                                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                                    with result_col1:
                                        st.metric("Shape", f"{shap_values.shape[0]} × {shap_values.shape[1]}")
                                    with result_col2:
                                        st.metric("Min Value", f"{shap_values.min():.6f}")
                                    with result_col3:
                                        st.metric("Max Value", f"{shap_values.max():.6f}")
                                    with result_col4:
                                        non_zero_pct = (shap_values != 0).sum() / shap_values.size * 100
                                        st.metric("Non-zero %", f"{non_zero_pct:.1f}%")
                                    
                                    if debug_level == "Full":
                                        # Show detailed statistics
                                        st.write("**📊 Detailed SHAP Statistics:**")
                                        stats_df = pd.DataFrame({
                                            'Metric': ['Mean', 'Std', 'Median', 'Q25', 'Q75', 'Sum'],
                                            'Value': [
                                                shap_values.mean(),
                                                shap_values.std(), 
                                                np.median(shap_values),
                                                np.percentile(shap_values, 25),
                                                np.percentile(shap_values, 75),
                                                shap_values.sum()
                                            ]
                                        })
                                        st.dataframe(stats_df, use_container_width=True)
                                        
                                        # Feature-wise statistics
                                        st.write("**📈 Per-Feature SHAP Statistics:**")
                                        feature_names = st.session_state.processed_data["feature_names"]
                                        feature_stats = []
                                        for i, fname in enumerate(feature_names):
                                            feature_shap = shap_values[:, i]
                                            feature_stats.append({
                                                'Feature': fname,
                                                'Mean |SHAP|': np.abs(feature_shap).mean(),
                                                'Max |SHAP|': np.abs(feature_shap).max(),
                                                'Non-zero %': (feature_shap != 0).sum() / len(feature_shap) * 100
                                            })
                                        
                                        feature_df = pd.DataFrame(feature_stats).sort_values('Mean |SHAP|', ascending=False)
                                        st.dataframe(feature_df.head(10), use_container_width=True)
                                        st.caption("Showing top 10 features by mean absolute SHAP value")
                                    
                                else:
                                    st.error("❌ Failed to calculate SHAP values")
                                    st.write("**Possible issues:**")
                                    st.write("• Model type not supported by SHAP")
                                    st.write("• Data format incompatible") 
                                    st.write("• Numerical instability")
                                    st.write("• Memory limitations")
                                    
                                    # Show model info for troubleshooting
                                    st.write("**🔧 Troubleshooting Info:**")
                                    st.code(f"""
Model: {type(model).__name__}
Has predict: {hasattr(model, 'predict')}
Has predict_proba: {hasattr(model, 'predict_proba')}
X_test dtype: {X_test.dtypes.unique()}
X_test shape: {X_test.shape}
Contains NaN: {X_test.isnull().any().any()}
Contains inf: {np.isinf(X_test.select_dtypes(include=[np.number])).any().any()}
                                    """)
                                    
                        except Exception as e:
                            st.error(f"❌ Debug failed: {str(e)}")
                            with st.expander("Full Error Details", expanded=False):
                                import traceback
                                st.code(traceback.format_exc())

        with tabs[3]:  # Error Analysis
            col1, col2 = st.columns([20, 1])
            with col1:
                st.subheader("Error Analysis")
            with col2:
                with st.popover("ℹ️"):
                    st.markdown(
                        """
                    **Error Analysis Overview**
                    
                    This section helps you understand where and why your model makes mistakes.
                    
                    **Available analyses:**
                    • **Regression**: Residual plots, prediction vs actual scatter plots
                    • **Classification**: Confusion matrices, error distribution by class
                    • **Learning Curves**: Training/validation performance over time
                    
                    **Why analyze errors?**
                    • Identify systematic biases
                    • Find problematic data ranges
                    • Detect overfitting/underfitting
                    • Improve model performance
                    """
                    )

            # Select model for error analysis
            model_for_error = st.selectbox(
                "Select Model for Error Analysis",
                list(training_results["models"].keys()),
                key="error_model",
            )

            if model_for_error:
                model_data = training_results["models"][model_for_error]

                if training_results["task_type"] == "regression":
                    # Regression error analysis
                    col1, col2 = st.columns([20, 1])
                    with col1:
                        st.markdown("#### Regression Error Analysis")
                    with col2:
                        with st.popover("ℹ️"):
                            st.markdown(
                                """
                            **Understanding Regression Error Plots**
                            
                            This visualization typically includes 4 subplots:
                            
                            **1. Residuals vs Predicted (Top Left)**
                            • **What it shows**: Prediction errors (residuals) plotted against predicted values
                            • **Ideal pattern**: Random scatter around y=0 line
                            • **Problems to spot**:
                              - Cone shape: Heteroscedasticity (variance changes with prediction level)
                              - Curves: Non-linear patterns the model missed
                              - Clusters: Distinct groups with different error patterns
                            
                            **2. Actual vs Predicted (Top Right)**
                            • **What it shows**: True values vs model predictions
                            • **Ideal pattern**: Points along the diagonal line (perfect predictions)
                            • **Problems to spot**:
                              - Systematic over/under-prediction in certain ranges
                              - Outliers far from the diagonal
                              - Poor fit in specific value ranges
                            
                            **3. Q-Q Plot (Bottom Left)**
                            • **What it shows**: Checks if residuals follow normal distribution
                            • **Ideal pattern**: Points along the straight reference line
                            • **Problems to spot**:
                              - S-curves: Heavy-tailed distributions
                              - Deviations at ends: Outliers or non-normality
                            
                            **4. Residuals Distribution (Bottom Right)**
                            • **What it shows**: Histogram of prediction errors
                            • **Ideal pattern**: Bell-shaped curve centered at 0
                            • **Problems to spot**:
                              - Skewed distributions
                              - Multiple peaks (bimodal)
                              - Very wide or narrow spread
                            
                            **Key Metrics to Check:**
                            • **Mean Absolute Error (MAE)**: Average prediction error
                            • **RMSE**: Penalizes large errors more
                            • **R²**: Proportion of variance explained
                            """
                            )

                    st.plotly_chart(
                        st.session_state.viz_engine.create_error_analysis(
                            model_data, st.session_state.processed_data
                        ),
                        use_container_width=True,
                    )
                else:
                    # Classification confusion matrices
                    col1, col2 = st.columns([20, 1])
                    with col1:
                        st.markdown("#### Classification Confusion Matrix")
                    with col2:
                        with st.popover("ℹ️"):
                            st.markdown(
                                """
                            **Understanding Confusion Matrices**
                            
                            A confusion matrix shows how your classifier performs for each class.
                            
                            **How to read:**
                            • **Rows**: Actual (true) classes
                            • **Columns**: Predicted classes
                            • **Diagonal**: Correct predictions (darker is better)
                            • **Off-diagonal**: Misclassifications
                            
                            **Key patterns to look for:**
                            
                            **1. Strong diagonal**: 
                            • Dark squares along diagonal = good performance
                            • Light diagonal = poor overall accuracy
                            
                            **2. Systematic errors**:
                            • Dark off-diagonal cells show consistent misclassifications
                            • Example: If row A, column B is dark, model often confuses A for B
                            
                            **3. Class imbalance effects**:
                            • Some rows much darker than others
                            • Model biased toward predicting common classes
                            
                            **4. Symmetric confusion**:
                            • If A→B and B→A are both high, classes are similar
                            • Consider feature engineering to distinguish them
                            
                            **Metrics from confusion matrix:**
                            • **Precision**: Of all predicted as X, how many were actually X?
                            • **Recall**: Of all actual X, how many were predicted as X?
                            • **F1-Score**: Balance between precision and recall
                            
                            **Common issues:**
                            • One class dominates predictions (column very dark)
                            • One class never predicted (column all light)
                            • Specific pairs frequently confused
                            """
                            )

                    st.plotly_chart(
                        st.session_state.viz_engine.create_confusion_matrices(
                            training_results
                        ),
                        use_container_width=True,
                    )

                # Learning curves
                if model_data.get("learning_curve"):
                    col1, col2 = st.columns([20, 1])
                    with col1:
                        st.subheader("Learning Curves")
                    with col2:
                        with st.popover("ℹ️"):
                            st.markdown(
                                """
                            **Understanding Learning Curves**
                            
                            Learning curves show how model performance changes with training set size.
                            
                            **Lines shown:**
                            • **Training score** (typically higher): Performance on training data
                            • **Validation score** (typically lower): Performance on held-out data
                            • **Shaded area**: Variance across cross-validation folds
                            
                            **How to interpret:**
                            
                            **1. High Bias (Underfitting)**
                            • Both curves converge at a low score
                            • Adding more data won't help much
                            • **Solution**: More complex model, more features
                            
                            **2. High Variance (Overfitting)**
                            • Large gap between training and validation scores
                            • Training score much higher than validation
                            • **Solution**: More data, regularization, simpler model
                            
                            **3. Good Fit**
                            • Curves converge at a high score
                            • Small gap between training and validation
                            • Model generalizes well
                            
                            **4. More data needed?**
                            • If validation score still increasing at right edge
                            • If gap between curves is narrowing
                            • More training data could improve performance
                            
                            **Key insights:**
                            • **Flat validation curve**: Model has learned all it can
                            • **Rising validation curve**: Model still improving with more data
                            • **Wide confidence bands**: High variance, unstable model
                            
                            **Typical progression:**
                            1. Small data: Both scores low (high bias)
                            2. Medium data: Gap appears (variance emerges)
                            3. Large data: Curves converge (if model capacity appropriate)
                            """
                            )
                    st.plotly_chart(
                        st.session_state.viz_engine.create_learning_curve(
                            model_data["learning_curve"]
                        ),
                        use_container_width=True,
                    )

        with tabs[4]:  # Custom Visualizations
            st.subheader("Custom Visualizations")

            # Statistical plots
            st.subheader("Statistical Analysis Plots")

            # Try to use cached results first
            stats = getattr(st.session_state, "statistical_results", None)

            if not stats:
                import pandas as pd
                from modules.statistical_analysis import StatisticalAnalysis as StatisticalAnalyzer

                # 1) Pull processed data container (whatever you saved during training)
                pdata = st.session_state.get("processed_data") or {}

                # 2) Build X with broad fallbacks (safe picker)
                x = None
                for key in ("X_train", "x_train", "features_df", "x", "data"):
                    cand = pdata.get(key, None)
                    if isinstance(cand, pd.DataFrame) and not cand.empty:
                        x = cand
                        break

                X = x

                # 3) Determine target name early if present
                target_name = (
                    pdata.get("target_name")
                    or pdata.get("target_column")
                    or (getattr(pdata.get("y_train"), "name", None))
                    or (getattr(pdata.get("y"), "name", None))
                )

                # 4) Build y with broad fallbacks (including from a full data frame)
                y = None
                for key in ("y_train", "y", "Y_train", "target"):
                    cand = pdata.get(key, None)
                    if cand is None:
                        continue
                    # accept Series, or a one-column DataFrame (squeeze it)
                    if isinstance(cand, pd.DataFrame):
                        if cand.shape[1] == 1:
                            cand = cand.iloc[:, 0]
                        else:
                            # if it's a multi-col DF, use target_name when available
                            if target_name and target_name in cand.columns:
                                cand = cand[target_name]
                            else:
                                continue
                    y = cand
                    break

                # if still None, try pdata["data"][target_name]
                if (
                    y is None
                    and isinstance(pdata.get("data"), pd.DataFrame)
                    and target_name
                    and target_name in pdata["data"].columns
                ):
                    y = pdata["data"][target_name]

                # 5) If X is a full DataFrame that still includes the target, drop it
                if isinstance(X, pd.DataFrame) and target_name and target_name in X.columns:
                    X = X.drop(columns=[target_name])

                # 6) Coerce X -> DataFrame if needed (and we have feature names)
                if X is not None and not isinstance(X, pd.DataFrame):
                    feat_cols = (
                        pdata.get("feature_names")
                        or pdata.get("features")
                        or [f"f{i}" for i in range(len(X[0]))]
                            if hasattr(X, "__len__") and len(X) and hasattr(X[0], "__len__")
                            else None
                    )
                    try:
                        X = pd.DataFrame(X, columns=feat_cols)
                    except Exception:
                        # fall back to a safest possible frame
                        X = pd.DataFrame(X)

                # 7) Only run if we actually have something to analyze
                if isinstance(X, pd.DataFrame) and (y is not None):
                    with st.spinner("Running statistical analysis..."):
                        cfg = getattr(st.session_state, "config", None) or Config()
                        analyzer = StatisticalAnalyzer(cfg)

                        # --- Build a DataFrame for stats that includes the target column ---
                        df_stats = X.copy()
                        if target_name and target_name not in df_stats.columns:
                            # coerce y -> Series and attach as column named target_name
                            if isinstance(y, pd.DataFrame):
                                y_series = y.iloc[:, 0]
                            elif isinstance(y, pd.Series):
                                y_series = y
                            else:
                                y_series = pd.Series(y)
                            y_series = y_series.reset_index(drop=True)
                            y_series.name = target_name

                            df_stats = df_stats.reset_index(drop=True)
                            if len(df_stats) == len(y_series):
                                df_stats[target_name] = y_series

                            # -- Prefer the widest API available in StatisticalAnalysis --
                            if hasattr(analyzer, "run_complete_analysis"):
                                try:
                                    # --- Decide task_type (prefer saved; otherwise infer) ---
                                    task_type = None
                                    try:
                                        task_type = (st.session_state.processed_data or {}).get("task_type")
                                    except Exception:
                                        pass

                                    # --- PLS options UI (always show) ---
                                    pls_target_name = None
                                    numeric_cols = []
                                    if isinstance(df_stats, pd.DataFrame):
                                        try:
                                            numeric_cols = df_stats.select_dtypes(include="number").columns.tolist()
                                        except Exception:
                                            numeric_cols = []

                                    with st.expander("⚙️ PLS options", expanded=True):
                                        if not numeric_cols:
                                            st.info("No numeric columns found, so PLS is disabled for this dataset.")
                                            n_pls = 1
                                        else:
                                            # prefer the inferred/previous target if it's still numeric
                                            default_idx = numeric_cols.index(target_name) if target_name in numeric_cols else 0
                                            pls_target_name = st.selectbox(
                                                "Target column for PLS",
                                                options=numeric_cols,
                                                index=default_idx,
                                                help="PLS finds components that best explain variance in X with respect to this target.",
                                                key="pls_target_select",
                                            )
                                            max_pls = min(8, len(numeric_cols))
                                            n_pls = st.slider(
                                                "Number of PLS components",
                                                min_value=1,
                                                max_value=max_pls,
                                                value=min(3, max_pls),
                                                step=1,
                                                help="More LVs can capture more structure; 2–3 is a good start.",
                                                key="pls_n_components",
                                            )

                                        # Keep a small summary visible so users see their choice persist
                                        if numeric_cols:
                                            st.caption(f"PLS target: {pls_target_name or target_name} | n_pls={n_pls}")

                                    # --- Always run the wide analysis with the correct task type ---
                                    stats = analyzer.run_complete_analysis(
                                        df_stats,
                                        target_column=pls_target_name or target_name,  # use UI choice if set
                                        task_type=task_type,
                                        n_components=n_pls,
                                    )

                                    # Cache for subsequent visits and for the Custom Visualizations section
                                    if isinstance(stats, dict) and stats:
                                        st.session_state.statistical_results = stats
                                    else:
                                        st.warning("Statistical analysis returned no results; PCA/PLS will be hidden.")

                                except TypeError:
                                    # Some versions may take the name positionally; try fallbacks
                                    try:
                                        stats = analyzer.run_complete_analysis(df_stats, target_name)
                                    except Exception:
                                        stats = None

                                    if stats is None:
                                        stats = getattr(analyzer, "results", None)

                                    if stats is None:
                                        if hasattr(analyzer, "run_full_suite"):
                                            stats = analyzer.run_full_suite(X, y, target_name=target_name)
                                        elif hasattr(analyzer, "analyze"):
                                            stats = analyzer.analyze(X, y)
                                        elif hasattr(analyzer, "run_all"):
                                            stats = analyzer.run_all(X, y)
                                        elif hasattr(analyzer, "run_basic"):
                                            stats = analyzer.run_basic(X, y)

                    # cache for subsequent visits
                    st.session_state.statistical_results = stats

            # 8) Plot (if we have results)
            if stats:
                stats_plots = st.session_state.viz_engine.create_statistical_plots(stats)

                # --- Optional: show PCA details (components & explained variance) ---
                # Components (loadings)
                if isinstance(stats, dict) and "pca_components" in stats:
                    with st.expander("🔎 PCA Components (feature loadings)", expanded=False):
                        st.dataframe(stats["pca_components"].round(3), use_container_width=True)

                # Explained variance ratio (EVR)
                if isinstance(stats, dict) and "pca_evr" in stats:
                    with st.expander("📈 PCA Explained Variance", expanded=False):
                        # bar_chart expects a DataFrame/Series indexed by labels
                        evr = stats["pca_evr"]
                        # if you prefer nicer labels like PC1, PC2, ...
                        evr.index = [f"PC{i+1}" for i in range(len(evr))]
                        st.bar_chart(evr)

            # --- PLS details (supervised) ---
            if isinstance(stats, dict) and "pls_x_weights" in stats:
                with st.expander("🧭 PLS Components (X weights)", expanded=False):
                    st.dataframe(stats["pls_x_weights"].round(3), use_container_width=True)

            if isinstance(stats, dict) and "pls_evr_x" in stats:
                with st.expander("🧭 PLS Explained Variance (X)", expanded=False):
                    evr_pls = stats["pls_evr_x"].copy()
                    # friendly labels already LV1..LVk
                    st.bar_chart(evr_pls)

                TITLES = {
                    "distributions": "Distribution Normality",
                    "correlation_heatmap": "Correlation Heatmap",
                    "pca_analysis": "PCA Variance Explained",
                    "clustering": "Clustering Analysis",
                    "vif": "VIF (Multicollinearity)",
                    "mutual_info": "Mutual Information",
                    "anova": "ANOVA Significance (-log10 p)",
                    "chi2": "Chi-square Significance (-log10 p)",
                    "heteroscedasticity": "Heteroscedasticity (Levene)",
                    "outliers": "Outliers (IQR)",
                    "ts_diagnostics": "Time-Series Diagnostics (Daily)",
                    "acf_daily": "Daily Autocorrelation (lags 1–10)",
                    "pls_x_weights": "PLS Components (X weights)",
                    "pls_evr_x": "PLS Explained Variance (X)",
                }

                HELP = {
                    "distributions": "**Normality tests** per feature (lower p ⇒ more non-normal); consider transforms/robust models.",
                    "correlation_heatmap": "**Pearson r** across features; |r| ≳ 0.8 suggests redundancy.",
                    "pca_analysis": "**PCA variance explained**; look for a 95% line to choose components.",
                    "clustering": "**k-means**: silhouette and inertia trend; elbow/silhouette guide k.",
                    "vif": "**Variance Inflation Factor** → >5 (orange) soft flag, >10 (red) strong multicollinearity.",
                    "mutual_info": "**Mutual Information** with target (captures non-linear dependence).",
                    "anova": "**One-way ANOVA** significance; taller bars (log10 p) mean stronger group separation.",
                    "chi2": "**Chi-square** significance for categorical features vs target (log10 p).",
                    "heteroscedasticity": "**Levene's** test p-values; small p ⇒ unequal variances (heteroscedasticity).",
                    "outliers": "**IQR method** outlier counts per feature.",
                    "ts_diagnostics": "**ADF/KPSS** p-values, inferred seasonal period, and strongest daily ACF lags.",
                    "acf_daily": "Bar chart of daily **ACF** for lags 1–10; spikes at 7/14 indicate weekly patterns.",
                    "pls_x_weights": (
                    "Partial Least Squares components (LVs) learned to best predict the chosen target. "
                    "Each value is a feature weight (loading) onto an LV; larger absolute weights mean the feature "
                    "contributes more to that LV."
                    ),
                    "pls_evr_x": (
                    "Heuristic share of X variance captured by each PLS latent variable (computed from scores). "
                    "Use it to gauge how many LVs you need."
                    ),
                }

                order = [
                    "distributions", "correlation_heatmap", "pca_analysis", "clustering",
                    "vif", "mutual_info", "anova", "chi2", "heteroscedasticity", "outliers",
                    "ts_diagnostics", "acf_daily",
                ]

                for key in order:
                    fig = stats_plots.get(key)
                    # Skip empty figures safely
                    if fig is None or (hasattr(fig, "data") and not getattr(fig, "data")):
                        continue
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader(TITLES.get(key, key.replace("_", " ").title()))
                    with col2:
                        st.caption(HELP.get(key, ""))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No statistical results available.")

            # Feature-prediction correlation
            st.subheader("Feature-Prediction Correlation")
            st.plotly_chart(
                st.session_state.viz_engine.create_feature_prediction_correlation(
                    training_results, st.session_state.processed_data
                ),
                use_container_width=True,
            )

        with tabs[5]:  # Bayesian Analysis
            st.subheader("Bayesian Model Analysis")
            
            # Filter for Bayesian models
            bayesian_models = {name: data for name, data in training_results["models"].items() 
                              if data.get("is_bayesian", False)}
            
            if bayesian_models:
                selected_bayesian = st.selectbox(
                    "Select Bayesian Model",
                    list(bayesian_models.keys())
                )
                
                model_data = bayesian_models[selected_bayesian]
                
                # Convergence diagnostics
                if "posterior_diagnostics" in model_data:
                    diag = model_data["posterior_diagnostics"]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Converged", 
                                 "✅ Yes" if diag["converged"] else "❌ No")
                    with col2:
                        st.metric("Max R-hat", f"{diag['max_rhat']:.3f}")
                    with col3:
                        st.metric("Posterior Samples", 
                                 f"{diag['n_samples']} × {diag['n_chains']}")
                    
                    if diag.get("warning"):
                        st.warning(diag["warning"])
                
                # Uncertainty visualization
                if model_data.get("supports_uncertainty") and BAYESIAN_MODELS_AVAILABLE and plot_posterior_predictive_check:
                    st.subheader("Prediction Uncertainty")
                    
                    # Create uncertainty plot
                    fig = plot_posterior_predictive_check(
                        model_data,
                        st.session_state.processed_data["X_test"],
                        st.session_state.processed_data["y_test"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance for Bayesian Ridge with ARD
                    if selected_bayesian == "Bayesian Ridge Regression":
                        model = model_data["model"]
                        if hasattr(model, 'get_feature_importance') and model.ard:
                            st.subheader("Automatic Relevance Determination")
                            
                            importance = model.get_feature_importance()
                            feature_names = st.session_state.processed_data["feature_names"]
                            
                            # Create importance plot
                            import plotly.graph_objects as go
                            fig_imp = go.Figure(go.Bar(
                                x=importance,
                                y=feature_names,
                                orientation='h'
                            ))
                            fig_imp.update_layout(
                                title="Feature Relevance (ARD)",
                                xaxis_title="Relevance Score",
                                yaxis_title="Features"
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)
                
                # Model comparison metrics
                st.subheader("Bayesian Model Comparison")
                
                if len(bayesian_models) > 1:
                    comparison_data = []
                    for name, data in bayesian_models.items():
                        row = {
                            "Model": name,
                            "R² Score": data["test_scores"].get("r2_score", 0),
                            "RMSE": data["test_scores"].get("rmse", 0),
                            "Training Time": data.get("training_time", 0)
                        }
                        
                        # Add convergence info
                        if "posterior_diagnostics" in data:
                            row["Converged"] = "✅" if data["posterior_diagnostics"]["converged"] else "❌"
                            row["R-hat"] = f"{data['posterior_diagnostics']['max_rhat']:.3f}"
                        
                        comparison_data.append(row)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                
                # Posterior distributions button
                if st.checkbox("Show Posterior Distributions"):
                    model = model_data["model"]
                    if hasattr(model, 'trace_'):
                        st.info("Posterior distribution visualization requires arviz. Showing summary statistics instead.")
                        
                        # Show basic stats
                        if hasattr(model, 'trace_'):
                            st.write("**Model Parameters:**")
                            st.write(f"- Number of chains: {model.n_chains}")
                            st.write(f"- Samples per chain: {model.n_samples}")
                            st.write(f"- Total posterior samples: {model.n_chains * model.n_samples}")
            
            else:
                st.info("No Bayesian models were trained. Select Bayesian models in the Model Training tab.")

        with tabs[6]:  # Multicollinearity tab
            st.subheader("Multicollinearity Analysis")
            
            if 'multicollinearity_analysis' in st.session_state.training_results:
                mc_analysis = st.session_state.training_results['multicollinearity_analysis']
                
                # Check if analysis actually contains data
                if mc_analysis and 'severity' in mc_analysis:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        severity = mc_analysis['severity']
                        severity_color = {"low": "🟢", "moderate": "🟡", "high": "🔴", "unknown": "⚪"}
                        st.metric("Severity", f"{severity_color.get(severity, '⚪')} {severity.title()}")
                    
                    with col2:
                        st.metric("High VIF Features", len(mc_analysis.get('high_vif_features', [])))
                    
                    with col3:
                        st.metric("Correlated Pairs", len(mc_analysis.get('correlated_pairs', [])))
                    
                    with col4:
                        st.metric("Recommended Action", mc_analysis.get('recommended_action', 'N/A').replace('_', ' ').title())
                    
                    # VIF Scores Table
                    if mc_analysis.get('vif_scores'):
                        st.subheader("VIF Scores")
                        vif_df = pd.DataFrame([
                            {"Feature": feat, "VIF": vif, "Status": "⚠️ High" if vif > 10 else "✅ Good"}
                            for feat, vif in mc_analysis['vif_scores'].items()
                        ]).sort_values('VIF', ascending=False)
                        
                        st.dataframe(vif_df, use_container_width=True)
                    
                    # Correlation Pairs
                    if mc_analysis.get('correlated_pairs'):
                        st.subheader("High Correlation Pairs")
                        corr_df = pd.DataFrame(mc_analysis['correlated_pairs'], 
                                            columns=['Feature 1', 'Feature 2', 'Correlation'])
                        corr_df['Correlation'] = corr_df['Correlation'].round(3)
                        st.dataframe(corr_df, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    if mc_analysis.get('severity') == 'high':
                        st.error("""
                        **High Multicollinearity Detected**
                        - Consider using Ridge or Lasso regression
                        - Remove redundant features using VIF > 10 threshold  
                        - Use PCA for dimensionality reduction
                        - SHAP interpretability may be less reliable
                        """)
                    elif mc_analysis.get('severity') == 'moderate':
                        st.warning("""
                        **Moderate Multicollinearity Detected**
                        - Ridge regression recommended over OLS
                        - Monitor VIF scores and correlations
                        - Tree-based models should handle this well
                        """)
                    else:
                        st.success("""
                        **Low Multicollinearity**
                        - All model types appropriate
                        - Feature interpretability should be reliable
                        - No special handling required
                        """)
                
                else:
                    st.info("Multicollinearity analysis was not performed. Enable it in Data Processing settings.")
            
            else:
                st.info("No multicollinearity analysis available. Enable multicollinearity handling in Data Processing.")
                
        # Fixed: Export options with proper attribute access
        st.divider()
        st.subheader("📤 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Generate Full Report", type="primary"):
                with st.spinner("Generating report..."):
                    report_path = st.session_state.report_generator.generate_report(
                        data_info={
                            "shape": st.session_state.data.shape,
                            "columns": st.session_state.data.columns.tolist(),
                            "target": st.session_state.processed_data.get(
                                "target_column", "Unknown"
                            ),
                        },
                        quality_report=getattr(
                            st.session_state, "quality_report", None
                        ),
                        processing_report=st.session_state.processed_data.get(
                            "processing_report", {}
                        ),
                        training_results=training_results,
                        statistical_results=getattr(
                            st.session_state, "statistical_results", None
                        ),
                        best_model_name=training_results["best_model"],
                    )
                    st.success(f"Report generated: {report_path}")

        with col2:
            if st.button("💾 Export Models"):
                with st.spinner("Exporting models..."):
                    export_path = st.session_state.export_manager.export_models(
                        training_results, format="joblib", include_preprocessing=True
                    )
                    st.success(f"Models exported to {export_path}")

        with col3:
            if st.button("📈 Export Visualizations"):
                viz_path = Path("aquavista_results/visualizations")
                viz_path.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save main visualizations
                figs_to_save = {
                    "performance_chart": st.session_state.viz_engine.create_performance_chart(
                        training_results
                    ),
                    "comparison_heatmap": st.session_state.viz_engine.create_model_comparison_heatmap(
                        training_results
                    ),
                    "feature_importance": st.session_state.viz_engine.create_aggregated_feature_importance(
                        training_results
                    ),
                }

                for name, fig in figs_to_save.items():
                    fig_path = viz_path / f"{name}_{timestamp}.html"
                    st.session_state.viz_engine.save_figure(fig, str(fig_path))

                st.success(f"Visualizations saved to {viz_path}")
    def run_docs(self):
        """
        Render the built-in documentation / user manual.
        Looks for markdown files in ./docs (next to aquavista_main.py).
        """
        import io, zipfile
        from pathlib import Path

        st.title("📘 AquaVista Documentation")

        # 1) Locate docs
        app_dir  = Path(__file__).parent
        docs_dir = app_dir / "docs"
        if not docs_dir.exists():
            st.info(
                "No docs folder was found. Create a folder named **docs** next to "
                "`aquavista_main.py` and place the markdown files there."
            )
            st.stop()

        # 2) Discover chapters (sorted for predictable order)
        md_files = sorted([p for p in docs_dir.glob("*.md")], key=lambda p: p.name.lower())
        if not md_files:
            st.warning("The docs folder is present, but no *.md files were found.")
            st.stop()

        # Optional: map file names -> friendly titles
        friendly = {
            "00_TOC.md": "Table of Contents",
            "01_QuickStart.md": "Quick Start",
            "02_Results_Analysis.md": "Results & Analysis",
            "03_Models_Training.md": "Models & Training",
            "04_Advanced_Topics.md": "Advanced Topics",
        }
        options = [friendly.get(p.name, p.name) for p in md_files]
        file_by_title = {friendly.get(p.name, p.name): p for p in md_files}

        # Remember selection across reruns
        if "doc_selection" not in st.session_state:
            st.session_state.doc_selection = options[0]

        sel = st.selectbox("Chapter", options, index=options.index(st.session_state.doc_selection))
        st.session_state.doc_selection = sel
        chosen_path = file_by_title[sel]

        # 3) Render the selected chapter
        try:
            text = chosen_path.read_text(encoding="utf-8")
        except Exception as e:
            st.error(f"Could not read {chosen_path.name}: {e}")
            st.stop()

        st.markdown(text, unsafe_allow_html=False)

        # 4) Downloads: current chapter (MD), all chapters (ZIP), combined manual (MD)
        st.divider()
        st.subheader("Downloads")

        # Current chapter download
        st.download_button(
            "⬇️ Download current chapter (.md)",
            data=text.encode("utf-8"),
            file_name=chosen_path.name,
            mime="text/markdown",
            use_container_width=True,
        )

        # Combined manual (simple concatenation)
        combined = "\n\n---\n\n".join(
            [p.read_text(encoding="utf-8") for p in md_files]
        )
        st.download_button(
            "⬇️ Download complete manual (.md)",
            data=combined.encode("utf-8"),
            file_name="AquaVista_Manual.md",
            mime="text/markdown",
            use_container_width=True,
        )

        # ZIP of all source files
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in md_files:
                zf.writestr(p.name, p.read_text(encoding="utf-8"))
        st.download_button(
            "⬇️ Download all chapters (.zip)",
            data=buf.getvalue(),
            file_name="AquaVista_Docs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    def run_predictions(self):
        """Run predictions on new data with proper decoding"""
        st.header("🔮 Make Predictions")

        is_valid, missing = self.validate_session_state(["training_results"])
        if not is_valid:
            self.show_prerequisite_warning("Predictions", missing)
            return

        # Select model for predictions
        model_names = list(st.session_state.training_results["models"].keys())
        selected_model = st.selectbox(
            "Select Model for Predictions",
            model_names,
            index=model_names.index(st.session_state.training_results["best_model"]),
        )

        st.divider()

        # Input method
        input_method = st.radio(
            "Input Method", ["Upload File", "Manual Entry", "Use Test Data"]
        )

        if input_method == "Manual Entry":
            st.subheader("Enter Feature Values")

            # Get feature names
            feature_names = st.session_state.processed_data["feature_names"]

            # Create input fields
            input_values = {}
            cols = st.columns(3)
            max_features = getattr(st.session_state.config, "max_manual_features", 20)
            for idx, feature in enumerate(feature_names[:max_features]):
                with cols[idx % 3]:
                    input_values[feature] = st.number_input(
                        feature, value=0.0, format="%.4f", key=f"input_{feature}"
                    )

            if st.button("Make Prediction"):
                try:
                    # Ensure all features are present
                    expected_features = st.session_state.processed_data["feature_names"]

                    # Create full feature vector with zeros for missing features
                    full_input = {}
                    for feature in expected_features:
                        full_input[feature] = input_values.get(feature, 0.0)

                    # Create DataFrame with correct feature order
                    pred_df = pd.DataFrame([full_input])[expected_features]

                    # Get model
                    model_data = st.session_state.training_results["models"][
                        selected_model
                    ]
                    model = model_data["model"]

                    # Make prediction
                    prediction = model.predict(pred_df)[0]

                    # Decode prediction if classifier with categorical target
                    if (st.session_state.processed_data.get(
                        "task_type"
                    ) == "classification"

                        and "target_encoder" in st.session_state.processed_data):

                        target_encoder = st.session_state.processed_data["target_encoder"]
                        
                        if target_encoder.encoder is not None:
                            # Decode to original label
                            decoded_prediction = target_encoder.inverse_transform(
                                [int(prediction)]
                            )[0]

                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(f"### Predicted Class: {decoded_prediction}")
                            with col2:
                                st.info(f"Encoded value: {int(prediction)}")

                            # Show class probabilities if available
                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba(pred_df)[0]
                                decoded_probs = target_encoder.decode_proba(
                                    probabilities.reshape(1, -1)
                                )

                                st.write("**Class Probabilities:**")
                                for class_name, probs in decoded_probs.items():
                                    prob_value = probs[0]
                                    st.progress(prob_value)
                                    st.caption(f"{class_name}: {prob_value:.2%}")
                        else:
                            # No encoding needed
                            st.success(f"### Prediction: {prediction}")
                    else:
                        # Regression prediction
                        st.success(f"### Prediction: {prediction:.4f}")

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}", exc_info=True)

        elif input_method == "Use Test Data":
            st.info("Using existing test data for predictions")

            if st.button("Show Test Set Predictions"):
                model_data = st.session_state.training_results["models"][selected_model]
                predictions = model_data["predictions"]
                y_test = st.session_state.processed_data["y_test"]

                # Decode if needed
                if (
                    st.session_state.processed_data.get("task_type") == "classification"
                    and "target_encoder" in st.session_state.processed_data
                ):


                    target_encoder = st.session_state.processed_data["target_encoder"]
                    if target_encoder.encoder is not None:
                        # Decode both actual and predicted
                        actual_decoded = target_encoder.inverse_transform(
                            y_test.astype(int)
                        )
                        predicted_decoded = target_encoder.inverse_transform(
                            predictions.astype(int)
                        )

                        results_df = pd.DataFrame(
                            {
                                "Actual": actual_decoded,
                                "Predicted": predicted_decoded,
                                "Correct": actual_decoded == predicted_decoded,
                            }
                        )
                    else:
                        results_df = pd.DataFrame(
                            {
                                "Actual": y_test,
                                "Predicted": predictions,
                                "Error": y_test - predictions,
                            }
                        )
                else:
                    # Regression results
                    results_df = pd.DataFrame(
                        {
                            "Actual": y_test,
                            "Predicted": predictions,
                            "Error": y_test - predictions,
                        }
                    )

                # Display results
                st.dataframe(arrow_safe(results_df.head(100)), use_container_width=True)


                # Summary statistics
                st.subheader("Prediction Summary")
                if "Error" in results_df.columns:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Error", f"{results_df['Error'].mean():.4f}")
                    with col2:
                        st.metric("Std Error", f"{results_df['Error'].std():.4f}")
                    with col3:
                        st.metric("Max Error", f"{results_df['Error'].abs().max():.4f}")
                else:
                    # Classification summary
                    accuracy = (results_df["Correct"].sum() / len(results_df)) * 100
                    st.metric("Accuracy", f"{accuracy:.1f}%")

    def create_graph_header_with_help(self, title: str, help_text: str):
        """Create a graph header with help popup"""
        col1, col2 = st.columns([20, 1])
        with col1:
            st.subheader(title)
        with col2:
            with st.popover("ℹ️"):
                st.markdown(help_text)

    def run_model_management(self):
        """Model management interface"""
        st.header("📦 Model Management")

        tabs = st.tabs(["📤 Save Models", "📥 Load Models", "📄 Compare Versions"])

        with tabs[0]:  # Save Models
            st.subheader("Save Trained Models")

            if hasattr(st.session_state, "training_results"):
                # List available models
                model_names = list(st.session_state.training_results["models"].keys())
                selected_model = st.selectbox(
                    "Select Model to Save", model_names, key="save_model_select"
                )

                model_description = st.text_area(
                    "Model Description",
                    placeholder="Enter a description for this model...",
                )

                if st.button("💾 Save Model"):
                    # Ensure directory exists
                    model_dir = Path("aquavista_results/models")
                    model_dir.mkdir(parents=True, exist_ok=True)

                    # Save model
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = f"{selected_model.replace(' ', '_')}_{timestamp}"
                    save_path = model_dir / f"{model_name}.joblib"

                    # Get model data
                    model_data = st.session_state.training_results["models"][
                        selected_model
                    ]

                    # Add metadata
                    model_data["metadata"] = {
                        "description": model_description,
                        "saved_at": datetime.now().isoformat(),
                        "performance": model_data["test_scores"],
                    }

                    st.session_state.model_manager.save_model(model_data, save_path)
                    st.success(f"Model saved to {save_path}")
            else:
                st.info("No trained models available to save")

        with tabs[1]:  # Load Models
            st.subheader("Load Saved Models")

            # List saved models
            model_dir = Path("aquavista_results/models")
            if model_dir.exists():
                model_files = list(model_dir.glob("*.joblib"))

                if model_files:
                    selected_file = st.selectbox(
                        "Select Model File", model_files, format_func=lambda x: x.name
                    )

                    if st.button("📥 Load Model"):
                        with st.spinner("Loading model..."):
                            loaded_model = st.session_state.model_manager.load_model(
                                selected_file
                            )
                            st.session_state.loaded_model = loaded_model
                            st.success("Model loaded successfully!")

                            # Display model info
                            if "metadata" in loaded_model:
                                st.write("**Model Information:**")
                                st.write(
                                    f"- Saved: {loaded_model['metadata'].get('saved_at', 'Unknown')}"
                                )
                                st.write(
                                    f"- Description: {loaded_model['metadata'].get('description', 'No description')}"
                                )

                                if "performance" in loaded_model["metadata"]:
                                    st.write("**Performance Metrics:**")
                                    for metric, value in loaded_model["metadata"][
                                        "performance"
                                    ].items():
                                        st.write(f"- {metric}: {value:.4f}")
                else:
                    st.info("No saved models found")
            else:
                st.info("Model directory not found")

        with tabs[2]:  # Compare Versions
            st.subheader("Compare Model Versions")

            # Model version comparison
            model_dir = Path("aquavista_results/models")
            if model_dir.exists():
                model_files = list(model_dir.glob("*.joblib"))
                if len(model_files) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        model1_file = st.selectbox(
                            "First Model",
                            model_files,
                            format_func=lambda x: x.name,
                            key="compare1",
                        )
                    with col2:
                        model2_file = st.selectbox(
                            "Second Model",
                            model_files,
                            format_func=lambda x: x.name,
                            key="compare2",
                        )

                    if st.button("Compare Models"):
                        try:
                            model1 = st.session_state.model_manager.load_model(
                                model1_file
                            )
                            model2 = st.session_state.model_manager.load_model(
                                model2_file
                            )

                            # Display comparison
                            st.write("### Performance Comparison")
                            if "metadata" in model1 and "metadata" in model2:
                                perf1 = model1["metadata"].get("performance", {})
                                perf2 = model2["metadata"].get("performance", {})

                                comparison_data = []
                                for metric in set(perf1.keys()) | set(perf2.keys()):
                                    comparison_data.append(
                                        {
                                            "Metric": metric,
                                            model1_file.name: perf1.get(metric, "N/A"),
                                            model2_file.name: perf2.get(metric, "N/A"),
                                            "Difference": (
                                                perf1.get(metric, 0)
                                                - perf2.get(metric, 0)
                                                if metric in perf1 and metric in perf2
                                                else "N/A"
                                            ),
                                        }
                                    )

                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(arrow_safe(comparison_df), use_container_width=True)

                        except Exception as e:
                            st.error(f"Error comparing models: {str(e)}")
                else:
                    st.info(
                        "Need at least 2 saved models to compare. Save more models to enable comparison."
                    )
            else:
                st.info("No saved models found for comparison.")

    def cleanup(self):
        """Cleanup resources on app shutdown - FIXED VERSION"""
        try:
            # Only try to access session state if we're in a valid Streamlit context
            from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
            
            # Check if we have a valid script run context
            ctx = get_script_run_ctx()
            if ctx is not None:
                # We're in a valid Streamlit context, safe to access session state
                try:
                    if hasattr(st.session_state, "performance_monitor") and st.session_state.performance_monitor:
                        st.session_state.performance_monitor.cleanup()
                    logger.info("AquaVista cleanup completed")
                except Exception as e:
                    logger.warning(f"Cleanup warning: {e}")
            else:
                # No valid context, skip session state cleanup
                logger.info("AquaVista cleanup skipped (no valid context)")
                
        except Exception as e:
            # Catch any cleanup errors to prevent them from propagating
            logger.error(f"Cleanup error: {e}")

    def __del__(self):
        """Destructor - SAFER VERSION"""
        try:
            # Only attempt cleanup if we're not already in a shutdown state
            import sys
            if not sys.is_finalizing():
                self.cleanup()
        except Exception:
            # Silently ignore destructor errors to prevent system instability
            pass
    def _show_page(self, title: str, page_fn):
        """Render a page and show a friendly error instead of a blank screen - IMPROVED"""
        import traceback
        
        try:
            st.subheader(title)
            page_fn()
        except Exception as e:
            # Show a user-friendly error instead of crashing
            st.error(f"**{title} Error**")
            
            with st.expander("Error Details", expanded=False):
                st.write(f"**Error Type:** {type(e).__name__}")
                st.write(f"**Error Message:** {str(e)}")
                st.code(traceback.format_exc(), language="python")
            
            # Offer recovery options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Retry", key=f"retry_{title}"):
                    st.rerun()
            with col2:
                if st.button("🏠 Go to Home", key=f"home_{title}"):
                    self.current_tab = TAB_HOME
                    st.rerun()
            
            # Log the error
            logger.error(f"{title} page error: {str(e)}", exc_info=True)

    # Add this at the very end of your aquavista_main.py file, replacing the existing run() method and main execution:

    def run(self):
        """Main application loop - FIXED VERSION"""
        print("DEBUG: Starting run() method")
        
        try:
            # Add debug output
            st.write("DEBUG: App is running!")  # This should appear if working
            
            # Ensure session state is initialized
            if not hasattr(st.session_state, "initialized"):
                print("DEBUG: Initializing session state")
                self.initialize_session_state()

            # Create sidebar with error handling
            print("DEBUG: Creating sidebar")
            try:
                current_tab = self.create_sidebar()
                print(f"DEBUG: Current tab selected: {current_tab}")
            except Exception as e:
                st.error(f"Sidebar error: {e}")
                print(f"ERROR: Sidebar error: {e}")
                current_tab = TAB_HOME  # fallback
                
            # Ensure we have a valid current tab
            if not hasattr(self, 'current_tab') or self.current_tab is None:
                self.current_tab = current_tab or TAB_HOME
                
            print(f"DEBUG: Routing to tab: {self.current_tab}")

            # Route to appropriate page with error boundaries
            try:
                if self.current_tab == TAB_HOME:
                    print("DEBUG: Running home page")
                    st.header("Home Page")  # Debug output
                    self.run_home_page()
                elif self.current_tab == TAB_PROCESS:
                    print("DEBUG: Running data processing")
                    self.run_data_processing()
                elif self.current_tab == TAB_TRAIN:
                    print("DEBUG: Running model training")
                    self.run_model_training()
                elif self.current_tab == TAB_RESULTS:
                    print("DEBUG: Running results analysis")
                    self.run_results_analysis()
                elif self.current_tab == TAB_PRED:
                    print("DEBUG: Running predictions")
                    self.run_predictions()
                elif self.current_tab == TAB_MANAGE:
                    print("DEBUG: Running model management")
                    self.run_model_management()
                else:  # TAB_DOCS
                    print("DEBUG: Running docs")
                    self.run_docs()
                    
            except Exception as e:
                st.error(f"Page rendering error: {e}")
                print(f"ERROR: Page rendering error: {e}")
                import traceback
                st.code(traceback.format_exc())
                
        except Exception as e:
            st.error(f"Critical application error: {e}")
            print(f"ERROR: Critical application error: {e}")
            import traceback
            st.code(traceback.format_exc())
            
        # Memory cleanup
        try:
            self.prevent_memory_leaks()
        except Exception as e:
            print(f"WARNING: Memory cleanup error: {e}")

    # ADD THE THREE METHODS HERE (after line 4437, before the class ends):

    def get_preset_explanation(self, preset: str, task_type: str, n_samples: int, n_features: int, selected_models: list) -> str:
        """Generate explanation for why specific models were recommended"""
        
        explanations = {
            "🎯 Recommended": f"""
            **Why these {len(selected_models)} models were selected:**
            
            This preset balances **accuracy**, **speed**, and **interpretability** for your dataset:
            - **Dataset size**: {n_samples:,} samples, {n_features} features
            - **Task type**: {task_type.title()}
            
            **Selected models and their strengths:**
            {self._format_model_explanations(selected_models, task_type)}
            
            **Why this combination works well:**
            - **Tree-based models** (Random Forest, XGBoost) handle non-linear patterns
            - **Linear models** (Ridge, Logistic) provide baseline and interpretability  
            - **Neural networks** capture complex interactions
            - **Bayesian models** quantify prediction uncertainty
            
            This diverse set ensures you find the best approach for your specific data patterns.
            """,
            
            "⚡ Fast Models": f"""
            **Speed-optimized selection for quick results:**
            
            These {len(selected_models)} models prioritize **training speed** over maximum accuracy:
            - **Training time**: < 30 seconds total
            - **Memory usage**: Low to moderate
            
            **Fast models included:**
            {self._format_model_explanations(selected_models, task_type)}
            
            **When to use:**
            - Quick data exploration and baseline establishment
            - Iterative model development
            - Resource-constrained environments
            - Proof of concept work
            
            **Trade-off**: May sacrifice 2-5% accuracy for 10x faster training.
            """,
            
            "🏆 High Accuracy": f"""
            **Maximum performance selection:**
            
            These {len(selected_models)} models prioritize **predictive accuracy** above all else:
            - **Training time**: 2-10 minutes (depending on data size)
            - **Memory usage**: High
            - **Complexity**: Advanced algorithms with extensive tuning
            
            **High-performance models:**
            {self._format_model_explanations(selected_models, task_type)}
            
            **Advanced techniques included:**
            - **Gradient boosting** with optimized hyperparameters
            - **Ensemble methods** that combine multiple models
            - **Neural networks** with architecture search
            - **Bayesian optimization** for hyperparameter tuning
            
            **Best for**: Production systems, competitions, critical predictions
            """,
            
            "🔬 All Models": f"""
            **Comprehensive model comparison:**
            
            Testing all {len(selected_models)} available models to find the absolute best for your data:
            - **Coverage**: Every algorithm type and variation
            - **Training time**: 5-30 minutes depending on data size
            - **Discovery**: May reveal unexpected high performers
            
            **Complete algorithm coverage:**
            {self._format_model_explanations(selected_models, task_type)}
            
            **Why test everything:**
            - **No assumptions** about what works best
            - **Algorithm discovery** - sometimes simple methods surprise
            - **Robustness** - compare across all model families
            - **Research-grade** comparison for academic or critical applications
            
            **Recommendation**: Great for final model selection or research projects.
            """
        }
        
        return explanations.get(preset, f"Selected {len(selected_models)} models from {preset}")

    def _format_model_explanations(self, models: list, task_type: str) -> str:
        """Format individual model explanations"""
        if not models:
            return "- No models selected"
        
        model_explanations = {
            # Tree-based
            'Random Forest': "**Random Forest** - Robust ensemble, handles missing values, built-in feature importance",
            'XGBoost': "**XGBoost** - Gradient boosting champion, excellent on structured data",
            'LightGBM': "**LightGBM** - Fast gradient boosting, great for large datasets", 
            'CatBoost': "**CatBoost** - Handles categorical features automatically",
            'Decision Tree': "**Decision Tree** - Highly interpretable, good baseline",
            'Extra Trees': "**Extra Trees** - Even more randomized than Random Forest",
            'Gradient Boosting': "**Gradient Boosting** - Sequential error correction",
            
            # Linear
            'Linear Regression': "**Linear Regression** - Simple, fast, interpretable baseline",
            'Ridge': "**Ridge Regression** - Linear with L2 regularization, prevents overfitting",
            'Lasso': "**Lasso** - Linear with L1 regularization, automatic feature selection",
            'Logistic Regression': "**Logistic Regression** - Linear classification, probabilistic output",
            'ElasticNet': "**ElasticNet** - Combines Ridge and Lasso regularization",
            
            # Advanced
            'Neural Network': "**Neural Network** - Deep learning, captures complex non-linear patterns",
            'SVM': "**Support Vector Machine** - Effective in high dimensions, kernel methods",
            'K-Neighbors': "**K-Neighbors** - Instance-based learning, good for local patterns",
            
            # Bayesian
            'Bayesian Ridge': "**Bayesian Ridge** - Uncertainty quantification, automatic relevance",
            'ARD Regression': "**ARD Regression** - Automatic feature relevance determination",
            'Bayesian Linear Regression': "**Bayesian Linear** - Full uncertainty quantification",
            'Bayesian Ridge Regression': "**Bayesian Ridge** - Principled regularization with uncertainty",
            
            # Robust
            'Huber': "**Huber** - Robust to outliers, combines L1/L2 loss",
            'RANSAC': "**RANSAC** - Extremely robust to outliers",
            'Theil-Sen': "**Theil-Sen** - Median-based robust regression",
            
            # Naive
            'Naive Bayes': "**Naive Bayes** - Fast probabilistic classifier, works with small data",
            
            # Ensemble
            'Voting Ensemble': "**Voting Ensemble** - Combines predictions from multiple models",
            'Stacking Ensemble': "**Stacking Ensemble** - Meta-learning from base model predictions"
        }
        
        explanations = []
        for model in models[:8]:  # Limit to first 8 to avoid overwhelming
            explanation = model_explanations.get(model, f"**{model}** - Advanced {task_type} algorithm")
            explanations.append(f"- {explanation}")
        
        if len(models) > 8:
            explanations.append(f"- ... and {len(models) - 8} more models")
        
        return "\n".join(explanations)

    def get_category_explanation(self, category: str, task_type: str) -> str:
        """Provide explanation for each model category"""
        explanations = {
            "Linear": "Simple, fast, and interpretable models. Great baselines that work well when relationships are roughly linear.",
            
            "Tree": "Decision tree-based models. Excellent at capturing non-linear patterns and interactions. Handle missing values well.",
            
            "Boosting": "Sequential learning algorithms that build strong predictors by combining weak learners. Often achieve highest accuracy.",
            
            "Neural": "Deep learning models that can capture very complex patterns. Require more data but can achieve excellent performance.",
            
            "SVM": "Support Vector Machines find optimal decision boundaries. Work well in high-dimensional spaces.",
            
            "Neighbors": "Instance-based learning that makes predictions based on similar examples. Good for local patterns.",
            
            "Bayesian": "Probabilistic models that quantify uncertainty in predictions. Provide confidence intervals and handle small data well.",
            
            "Robust": "Models designed to handle outliers and noisy data. Use when your dataset has data quality issues.",
            
            "Gaussian Process": "Non-parametric Bayesian approach. Excellent for small datasets and uncertainty quantification.",
            
            "Bayesian Models": "Advanced Bayesian inference models with full uncertainty quantification. Ideal for critical decisions requiring confidence estimates."
        }
        
        return explanations.get(category, f"Specialized {task_type} algorithms in the {category} family.")


# FIXED Main execution
if __name__ == "__main__":
    print("DEBUG: Starting AquaVista app")
    try:
        app = AquaVistaApp()
        print("DEBUG: App initialized, calling run()")
        app.run()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Show error in Streamlit too
        st.error(f"Failed to start AquaVista: {e}")
        st.code(traceback.format_exc())
        st.code(traceback.format_exc())
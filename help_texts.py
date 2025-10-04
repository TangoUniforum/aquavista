"""
Help text definitions for AquaVista v6.0
=======================================
Comprehensive help texts for all UI elements
"""

HELP_TEXTS = {
    # Data Processing Help
    'target_column': """
        **Target Variable Selection**
        
        The target variable is what you want to predict. Choose wisely as this determines:
        
        • **Task Type**: Classification (categories) or Regression (continuous values)
        • **Model Selection**: Different models work better for different tasks
        • **Evaluation Metrics**: Accuracy for classification, R² for regression
        
        **Tips:**
        • For water quality, common targets include: pH levels, contamination presence, quality grade
        • Ensure your target has sufficient variation
        • Check for class imbalance in classification tasks
    """,
    
    'feature_columns': """
        **Feature Selection Guide**
        
        Features are the inputs your model uses to make predictions.
        
        **Good features:**
        • Have predictive power (correlate with target)
        • Are available at prediction time
        • Don't leak information from the future
        • Have reasonable data quality
        
        **Avoid features that:**
        • Are >50% missing
        • Have near-zero variance
        • Are highly correlated with each other (>0.95)
        • Are IDs or timestamps (unless time-aware)
    """,
    
    'missing_strategy': """
        **Missing Value Strategies**
        
        • **Auto**: Intelligent selection based on data type and distribution
        • **Mean**: Replace with average (good for normal distributions)
        • **Median**: Replace with middle value (robust to outliers)
        • **Mode**: Most frequent value (good for categorical)
        • **Forward Fill**: Use previous value (good for time series)
        • **Interpolate**: Estimate between known values
        • **Drop**: Remove rows with missing values (careful with data loss)
        
        **Rule of thumb**: Use median for skewed numeric data, mode for categorical
    """,
    
    'scaling_method': """
        **Feature Scaling Methods**
        
        • **Auto**: Best method selected per model type
        • **Standard**: Zero mean, unit variance (good for most algorithms)
        • **MinMax**: Scale to [0,1] range (good for neural networks)
        • **Robust**: Uses median/IQR, resistant to outliers
        • **None**: No scaling (tree-based models don't need it)
        
        **When to use:**
        • Neural Networks, SVM: Always scale
        • Linear models: Usually benefit from scaling
        • Tree-based: Scaling not required
    """,
    
    'encoding_method': """
        **Categorical Encoding Methods**
        
        • **Auto**: Smart selection based on cardinality
        • **OneHot**: Create binary columns (good for <10 categories)
        • **Target**: Encode based on target mean (prevents overfitting)
        • **Ordinal**: Integer encoding (use when order matters)
        
        **Tips:**
        • High cardinality (>50 categories): Use target encoding
        • Low cardinality (<10): Use one-hot encoding
        • Ordinal data: Use ordinal encoding
    """,
    
    'cv_folds': """
        **Cross-Validation Folds**
        
        Number of data splits for robust model evaluation.
        
        • **2-3 folds**: Very large datasets or slow models
        • **5 folds**: Standard choice (80/20 splits)
        • **10 folds**: Small datasets (<1000 samples)
        
        **Trade-offs:**
        • More folds = More robust evaluation
        • More folds = Longer training time
        • Less data per fold with more splits
    """,
    
    'tuning_budget': """
        **Hyperparameter Tuning Budget**
        
        Controls the extent of hyperparameter search:
        
        • **Quick**: Basic grid search, ~5 combinations
        • **Standard**: Moderate search, ~20 combinations
        • **Extensive**: Thorough search, ~50+ combinations
        
        **Time estimates:**
        • Quick: 1-2x base training time
        • Standard: 3-5x base training time  
        • Extensive: 10x+ base training time
    """,
    
    'ensemble_methods': """
        **Ensemble Method Types**
        
        Combine multiple models for better performance:
        
        • **Voting**: Average predictions (regression) or majority vote (classification)
        • **Averaging**: Simple mean of predictions (regression only)
        • **Stacking**: Train a meta-model on base model predictions
        
        **When to use:**
        • Voting: When models have similar performance
        • Stacking: When you have diverse model types
        • Generally improves robustness and reduces overfitting
    """
}

def get_help_text(key: str) -> str:
    """Get help text for a given key"""
    return HELP_TEXTS.get(key, "No help text available for this item.")

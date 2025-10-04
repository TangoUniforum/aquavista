"""
Bayesian Models Module for AquaVista v7.0
Implements Bayesian regression and classification with uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pymc as pm
import arviz as az
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import logging
import warnings

# Suppress PyTensor compiler warnings
import os
os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,cxx='

logger = logging.getLogger(__name__)

def _ensure_numpy_array(y):
    """Convert pandas Series to numpy array if necessary"""
    if hasattr(y, 'values'):  # pandas Series or DataFrame
        return y.values
    return y

class BayesianLinearRegression(BaseEstimator, RegressorMixin):
    """Bayesian Linear Regression with uncertainty quantification"""
    
    def __init__(self, 
                 n_samples: int = 1000,  # Reduced from 2000 for faster sampling
                 n_chains: int = 2,      # Reduced from 4 for faster sampling
                 target_accept: float = 0.8,  # Reduced from 0.9 for faster sampling
                 random_state: int = 42):
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_state = random_state
        self.model_ = None
        self.trace_ = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.posterior_samples_ = None
        
    def fit(self, X, y):
        """Fit Bayesian Linear Regression model"""
        # Standardize data
        X_scaled = self.scaler_X.fit_transform(X)
        # Fix: Handle pandas Series
        y_array = _ensure_numpy_array(y)
        y_scaled = self.scaler_y.fit_transform(y_array.reshape(-1, 1)).ravel()
        
        n_features = X_scaled.shape[1]
        
        with pm.Model() as self.model_:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            betas = pm.Normal('betas', mu=0, sigma=10, shape=n_features)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear model
            mu = alpha + pm.math.dot(X_scaled, betas)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_scaled)
            
            # Sample from posterior with optimizations
            try:
                self.trace_ = pm.sample(
                    draws=self.n_samples,
                    tune=min(1000, self.n_samples),  # Limit tuning steps
                    chains=self.n_chains,
                    target_accept=self.target_accept,
                    random_seed=self.random_state,
                    progressbar=False,
                    return_inferencedata=True,
                    cores=1  # Use single core to avoid multiprocessing issues
                )
                
                # Store posterior samples for prediction
                self.posterior_samples_ = {
                    'alpha': self.trace_.posterior['alpha'].values.flatten(),
                    'betas': self.trace_.posterior['betas'].values.reshape(-1, n_features),
                    'sigma': self.trace_.posterior['sigma'].values.flatten()
                }
                
            except Exception as e:
                logger.warning(f"Sampling failed, using MAP estimate: {e}")
                # Fall back to MAP estimate if sampling fails
                map_estimate = pm.find_MAP()
                self.posterior_samples_ = {
                    'alpha': np.array([map_estimate['alpha']]),
                    'betas': map_estimate['betas'].reshape(1, -1),
                    'sigma': np.array([map_estimate['sigma']])
                }
        
        return self
    
    def predict(self, X, return_std=False):
        """Make predictions with optional uncertainty"""
        if self.posterior_samples_ is None:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler_X.transform(X)
        
        # Calculate predictions from posterior samples
        alpha_samples = self.posterior_samples_['alpha']
        betas_samples = self.posterior_samples_['betas']
        
        # Predictions for each posterior sample
        y_pred_samples = alpha_samples[:, np.newaxis] + np.dot(betas_samples, X_scaled.T)
        
        # Mean prediction
        y_pred_scaled = y_pred_samples.mean(axis=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        if return_std:
            # Include both parameter and noise uncertainty
            sigma_samples = self.posterior_samples_['sigma']
            # Add noise uncertainty
            total_variance = y_pred_samples.var(axis=0) + (sigma_samples**2).mean()
            y_std = np.sqrt(total_variance) * self.scaler_y.scale_[0]
            return y_pred, y_std
        
        return y_pred
    
    def predict_proba_intervals(self, X, alpha=0.05):
        """Get prediction intervals"""
        X_scaled = self.scaler_X.transform(X)
        
        # Calculate predictions from posterior samples
        alpha_samples = self.posterior_samples_['alpha']
        betas_samples = self.posterior_samples_['betas']
        sigma_samples = self.posterior_samples_['sigma']
        
        # Predictions for each posterior sample
        y_pred_samples = alpha_samples[:, np.newaxis] + np.dot(betas_samples, X_scaled.T)
        
        # Add noise uncertainty
        n_samples, n_points = y_pred_samples.shape
        noise = np.random.normal(0, 1, (n_samples, n_points)) * sigma_samples[:, np.newaxis]
        y_pred_with_noise = y_pred_samples + noise
        
        # Calculate intervals
        lower = np.percentile(y_pred_with_noise, alpha/2 * 100, axis=0)
        upper = np.percentile(y_pred_with_noise, (1 - alpha/2) * 100, axis=0)
        
        # Inverse transform
        lower = self.scaler_y.inverse_transform(lower.reshape(-1, 1)).ravel()
        upper = self.scaler_y.inverse_transform(upper.reshape(-1, 1)).ravel()
        
        return lower, upper


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    """Bayesian Logistic Regression for binary classification"""
    
    def __init__(self,
                 n_samples: int = 1000,
                 n_chains: int = 2,
                 target_accept: float = 0.8,
                 random_state: int = 42):
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_state = random_state
        self.model_ = None
        self.trace_ = None
        self.scaler_ = StandardScaler()
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit Bayesian Logistic Regression"""
        # Handle classes
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("BayesianLogisticRegression only supports binary classification")
        
        # Convert to 0/1
        y_binary = (y == self.classes_[1]).astype(int)
        
        # Standardize features
        X_scaled = self.scaler_.fit_transform(X)
        n_features = X_scaled.shape[1]
        
        with pm.Model() as self.model_:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            betas = pm.Normal('betas', mu=0, sigma=10, shape=n_features)
            
            # Logistic model
            logit_p = alpha + pm.math.dot(X_scaled, betas)
            
            # Likelihood
            y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y_binary)
            
            # Sample
            try:
                self.trace_ = pm.sample(
                    draws=self.n_samples,
                    tune=min(1000, self.n_samples),
                    chains=self.n_chains,
                    target_accept=self.target_accept,
                    random_seed=self.random_state,
                    progressbar=False,
                    return_inferencedata=True,
                    cores=1
                )
            except Exception as e:
                logger.warning(f"Sampling failed, using MAP estimate: {e}")
                # Fall back to MAP estimate
                map_estimate = pm.find_MAP()
                # Create mock trace
                self.trace_ = type('MockTrace', (), {
                    'posterior': {
                        'alpha': type('MockData', (), {'values': np.array([[map_estimate['alpha']]])})(),
                        'betas': type('MockData', (), {'values': map_estimate['betas'].reshape(1, 1, -1)})()
                    }
                })()
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X_scaled = self.scaler_.transform(X)
        
        # Get posterior samples
        alpha_samples = self.trace_.posterior['alpha'].values.flatten()
        betas_samples = self.trace_.posterior['betas'].values.reshape(-1, X_scaled.shape[1])
        
        # Calculate probabilities for each posterior sample
        logits = alpha_samples[:, np.newaxis] + np.dot(betas_samples, X_scaled.T)
        probs = 1 / (1 + np.exp(-logits))
        
        # Average over samples
        prob_class_1 = probs.mean(axis=0)
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict(self, X):
        """Predict classes"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_proba_intervals(self, X, alpha=0.05):
        """Get prediction probability intervals"""
        X_scaled = self.scaler_.transform(X)
        
        # Get posterior samples
        alpha_samples = self.trace_.posterior['alpha'].values.flatten()
        betas_samples = self.trace_.posterior['betas'].values.reshape(-1, X_scaled.shape[1])
        
        # Calculate probabilities
        logits = alpha_samples[:, np.newaxis] + np.dot(betas_samples, X_scaled.T)
        probs = 1 / (1 + np.exp(-logits))
        
        # Calculate intervals
        lower = np.percentile(probs, alpha/2 * 100, axis=0)
        upper = np.percentile(probs, (1 - alpha/2) * 100, axis=0)
        
        return lower, upper


class BayesianRidge(BaseEstimator, RegressorMixin):
    """Bayesian Ridge Regression with automatic relevance determination"""
    
    def __init__(self,
                 n_samples: int = 1000,
                 n_chains: int = 2,
                 ard: bool = True,
                 random_state: int = 42):
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.ard = ard  # Automatic Relevance Determination
        self.random_state = random_state
        self.model_ = None
        self.trace_ = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X, y):
        """Fit Bayesian Ridge Regression"""
        X_scaled = self.scaler_X.fit_transform(X)
        # Fix: Handle pandas Series
        y_array = _ensure_numpy_array(y)
        y_scaled = self.scaler_y.fit_transform(y_array.reshape(-1, 1)).ravel()
        n_features = X_scaled.shape[1]
        
        with pm.Model() as self.model_:
            # Hyperpriors
            if self.ard:
                # Different precision for each feature
                lambda_reg = pm.Gamma('lambda_reg', alpha=1e-3, beta=1e-3, shape=n_features)
            else:
                # Single precision for all features
                lambda_reg = pm.Gamma('lambda_reg', alpha=1e-3, beta=1e-3)
            
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            
            if self.ard:
                betas = pm.Normal('betas', mu=0, tau=lambda_reg, shape=n_features)
            else:
                betas = pm.Normal('betas', mu=0, tau=lambda_reg, shape=n_features)
            
            # Noise precision
            tau_noise = pm.Gamma('tau_noise', alpha=1e-3, beta=1e-3)
            
            # Linear model
            mu = alpha + pm.math.dot(X_scaled, betas)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, tau=tau_noise, observed=y_scaled)
            
            # Sample
            try:
                self.trace_ = pm.sample(
                    draws=self.n_samples,
                    tune=min(1000, self.n_samples),
                    chains=self.n_chains,
                    random_seed=self.random_state,
                    progressbar=False,
                    return_inferencedata=True,
                    cores=1
                )
            except Exception as e:
                logger.warning(f"Sampling failed, using MAP estimate: {e}")
                # Fall back to MAP estimate
                map_estimate = pm.find_MAP()
                # Create mock trace
                self.trace_ = type('MockTrace', (), {
                    'posterior': {
                        'alpha': type('MockData', (), {'values': np.array([[map_estimate['alpha']]])})(),
                        'betas': type('MockData', (), {'values': map_estimate['betas'].reshape(1, 1, -1)})(),
                        'lambda_reg': type('MockData', (), {
                            'values': map_estimate['lambda_reg'].reshape(1, 1, -1) if self.ard else np.array([[map_estimate['lambda_reg']]]),
                            'mean': lambda dim=None: map_estimate['lambda_reg']
                        })()
                    }
                })()
        
        return self
    
    def predict(self, X, return_std=False):
        """Make predictions"""
        X_scaled = self.scaler_X.transform(X)
        
        # Get posterior samples
        alpha_samples = self.trace_.posterior['alpha'].values.flatten()
        betas_samples = self.trace_.posterior['betas'].values.reshape(-1, X_scaled.shape[1])
        
        # Calculate predictions
        y_pred_samples = alpha_samples[:, np.newaxis] + np.dot(betas_samples, X_scaled.T)
        
        # Average over samples
        y_pred_scaled = y_pred_samples.mean(axis=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        if return_std:
            y_std_scaled = y_pred_samples.std(axis=0)
            y_std = y_std_scaled * self.scaler_y.scale_
            return y_pred, y_std
        
        return y_pred
    
    def get_feature_importance(self):
        """Get feature importance from ARD"""
        if not self.ard:
            raise ValueError("Feature importance only available with ARD=True")
        
        # Get inverse of lambda (variance of weights)
        lambda_values = self.trace_.posterior['lambda_reg'].mean(dim=['chain', 'draw']).values
        importance = 1.0 / lambda_values
        
        return importance / importance.sum()


class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """Gaussian Process Regression - Simplified for faster performance"""
    
    def __init__(self,
                 kernel: str = 'rbf',
                 n_samples: int = 500,  # Reduced for GP
                 n_chains: int = 2,
                 random_state: int = 42):
        self.kernel = kernel
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.random_state = random_state
        self.model_ = None
        self.trace_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._length_scale = None
        self._eta = None
        self._sigma = None
        
    def fit(self, X, y):
        """Fit Gaussian Process"""
        # Limit data size for GP
        max_samples = 200  # GPs are O(n^3), so limit samples
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X_subset = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
            y_subset = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
        else:
            X_subset = X
            y_subset = y
            
        self.X_train_ = self.scaler_X.fit_transform(X_subset)
        # Fix: Handle pandas Series
        y_array = _ensure_numpy_array(y_subset)
        self.y_train_ = self.scaler_y.fit_transform(y_array.reshape(-1, 1)).ravel()
        
        # Use simple sklearn GP for speed
        try:
            # Fit hyperparameters using simple optimization
            from sklearn.gaussian_process import GaussianProcessRegressor as SklearnGP
            from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
            
            if self.kernel == 'rbf':
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            else:
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
                
            self.gp_sklearn = SklearnGP(kernel=kernel, alpha=0.1, random_state=self.random_state)
            self.gp_sklearn.fit(self.X_train_, self.y_train_)
            
            # Store fitted parameters
            self._fitted = True
            
        except Exception as e:
            logger.warning(f"GP fitting failed: {e}")
            # Fall back to simple prediction
            self._fitted = False
            
        return self
    
    def predict(self, X, return_std=False):
        """Make predictions"""
        X_test = self.scaler_X.transform(X)
        
        if hasattr(self, 'gp_sklearn') and self._fitted:
            # Use sklearn GP for predictions
            if return_std:
                y_pred_scaled, y_std_scaled = self.gp_sklearn.predict(X_test, return_std=True)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                y_std = y_std_scaled * self.scaler_y.scale_[0]
                return y_pred, y_std
            else:
                y_pred_scaled = self.gp_sklearn.predict(X_test)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                return y_pred
        else:
            # Simple fallback - return mean
            y_pred = np.full(len(X_test), self.scaler_y.inverse_transform([[self.y_train_.mean()]])[0, 0])
            if return_std:
                y_std = np.full(len(X_test), self.y_train_.std() * self.scaler_y.scale_[0])
                return y_pred, y_std
            return y_pred


# Utility functions for Bayesian model analysis
def calculate_waic(trace, model):
    """Calculate WAIC (Widely Applicable Information Criterion)"""
    try:
        with model:
            waic = az.waic(trace)
        return waic
    except:
        return None


def calculate_loo(trace, model):
    """Calculate LOO (Leave-One-Out cross-validation)"""
    try:
        with model:
            loo = az.loo(trace)
        return loo
    except:
        return None


def plot_posterior_predictive_check(model_data, X_test, y_test):
    """Create posterior predictive check plot"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    model = model_data['model']
    
    # Ensure numpy arrays
    y_test_array = _ensure_numpy_array(y_test)
    
    # Get predictions with uncertainty
    try:
        if hasattr(model, 'predict'):
            if hasattr(model, 'predict_proba_intervals'):
                y_pred = model.predict(X_test)
                lower, upper = model.predict_proba_intervals(X_test)
                # Calculate std from intervals
                y_std = (upper - lower) / 4  # Approximate std from 95% CI
            else:
                # Try to get predictions with std
                try:
                    y_pred, y_std = model.predict(X_test, return_std=True)
                    # Calculate intervals from std
                    lower = y_pred - 2 * y_std
                    upper = y_pred + 2 * y_std
                except:
                    y_pred = model.predict(X_test)
                    y_std = np.zeros_like(y_pred)
                    lower, upper = y_pred, y_pred
        else:
            # Fallback
            y_pred = np.zeros_like(y_test_array)
            y_std = np.zeros_like(y_test_array)
            lower, upper = y_pred, y_pred
            
    except Exception as e:
        logger.warning(f"Error in prediction: {e}")
        y_pred = np.zeros_like(y_test_array)
        y_std = np.zeros_like(y_test_array)
        lower, upper = y_pred, y_pred
    
    # Create figure
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Predictions vs Actual', 'Prediction Intervals'))
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=y_test_array, y=y_pred, mode='markers',
                   name='Predictions',
                   error_y=dict(type='data', array=y_std, visible=True)),
        row=1, col=1
    )
    
    # Perfect prediction line
    min_val, max_val = min(y_test_array.min(), y_pred.min()), max(y_test_array.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                   mode='lines', name='Perfect Prediction',
                   line=dict(dash='dash')),
        row=1, col=1
    )
    
    # Prediction intervals
    sorted_idx = np.argsort(y_test_array)
    fig.add_trace(
        go.Scatter(x=y_test_array[sorted_idx], y=y_pred[sorted_idx],
                   mode='lines+markers', name='Predictions'),
        row=1, col=2
    )
    
    # Add confidence bands
    fig.add_trace(
        go.Scatter(x=y_test_array[sorted_idx], y=upper[sorted_idx],
                   mode='lines', name='Upper 95% CI',
                   line=dict(width=0),
                   showlegend=False),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=y_test_array[sorted_idx], y=lower[sorted_idx],
                   mode='lines', name='Lower 95% CI',
                   line=dict(width=0),
                   fill='tonexty',
                   fillcolor='rgba(0,100,80,0.2)',
                   showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(title="Bayesian Model Uncertainty Quantification")
    fig.update_xaxes(title_text="Actual", row=1, col=1)
    fig.update_yaxes(title_text="Predicted", row=1, col=1)
    fig.update_xaxes(title_text="Actual", row=1, col=2)
    fig.update_yaxes(title_text="Predicted", row=1, col=2)
    
    return fig
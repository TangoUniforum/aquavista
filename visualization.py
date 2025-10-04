"""
Visualization Engine Module for AquaVista v6.0
=============================================
Creates all visualizations for the platform using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import warnings
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Import custom modules
from modules.config import Config
from modules.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class VisualizationEngine:
    """Creates visualizations for AquaVista platform"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.default_colorscale = px.colors.sequential.Viridis
        self.categorical_colors = px.colors.qualitative.Set3
        
        # Set default theme
        self.template = 'plotly_dark' if self._is_dark_mode() else 'plotly_white'
        
    def _is_dark_mode(self) -> bool:
        """Check if dark mode is preferred"""
        # Could be extended to check system preferences or config
        return False
    
    def get_layout(self, title: str = "", xaxis_title: str = "", 
                   yaxis_title: str = "", height: int = 500,
                   showlegend: bool = True) -> Dict[str, Any]:
        """Get consistent layout for all plots"""
        return {
            'title': {
                'text': title,
                'font': {'size': 20}
            },
            'xaxis': {
                'title': xaxis_title,
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': 'rgba(128, 128, 128, 0.2)'
            },
            'yaxis': {
                'title': yaxis_title,
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': 'rgba(128, 128, 128, 0.2)'
            },
            'height': height,
            'template': self.template,
            'showlegend': showlegend,
            'hovermode': 'x unified',
            'margin': {'l': 50, 'r': 20, 't': 60, 'b': 50}
        }
    
    def create_performance_summary(self, training_results: Dict[str, Any]) -> pd.DataFrame:
        """Create performance summary dataframe"""
        
        data = []
        primary_metric = training_results['primary_metric']
        
        # Track best values for each metric
        best_values = {}
        
        for model_name, model_data in training_results['models'].items():
            row = {
                'Model': model_name,
                'Training Time (s)': round(model_data['training_time'], 2)
            }
            
            # Add test scores
            for metric, value in model_data['test_scores'].items():
                col_name = f'Test {metric.upper()}'
                row[col_name] = round(value, 4)
                
                # Track best value
                if col_name not in best_values:
                    best_values[col_name] = {'value': value, 'model': model_name}
                else:
                    # For time and error metrics, lower is better
                    if 'TIME' in col_name.upper() or any(m in col_name.upper() for m in ['MSE', 'RMSE', 'MAE']):
                        if value < best_values[col_name]['value']:
                            best_values[col_name] = {'value': value, 'model': model_name}
                    else:
                        if value > best_values[col_name]['value']:
                            best_values[col_name] = {'value': value, 'model': model_name}
            
            # Add CV scores if available
            if model_data.get('cv_scores'):
                cv_mean_col = 'CV Mean'
                cv_std_col = 'CV Std'
                row[cv_mean_col] = round(model_data['cv_scores']['mean'], 4)
                row[cv_std_col] = round(model_data['cv_scores']['std'], 4)
                
                # Track best CV mean
                if cv_mean_col not in best_values:
                    best_values[cv_mean_col] = {'value': row[cv_mean_col], 'model': model_name}
                elif row[cv_mean_col] > best_values[cv_mean_col]['value']:
                    best_values[cv_mean_col] = {'value': row[cv_mean_col], 'model': model_name}
            
            data.append(row)
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Sort by primary metric
        sort_col = f'Test {primary_metric.upper()}'
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False)
        
        # Store best values info in dataframe attributes for styling
        df.attrs['best_values'] = best_values
        
        return df
    
    def create_performance_chart(self, training_results: Dict[str, Any]) -> go.Figure:
        """Create model performance comparison chart"""
        
        models = []
        scores = []
        colors = []
        texts = []
        
        primary_metric = training_results['primary_metric']
        best_model = training_results['best_model']
        baseline = training_results['baseline_score']
        
        for model_name, model_data in training_results['models'].items():
            score = model_data['test_scores'][primary_metric]
            
            # Handle negative scores
            if score < 0:
                # Skip negative scores in bar chart
                continue
            
            models.append(model_name)
            scores.append(score)
            colors.append('gold' if model_name == best_model else 'lightblue')
            texts.append(f'{score:.4f}')
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        models = [models[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        texts = [texts[i] for i in sorted_indices]
        
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=models,
            y=scores,
            marker_color=colors,
            text=texts,
            textposition='outside',
            name='Score',
            hovertemplate='Model: %{x}<br>Score: %{y:.4f}<extra></extra>'
        ))
        
        # Add baseline line (only if positive)
        if baseline > 0:
            fig.add_hline(
                y=baseline,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Baseline: {baseline:.4f}"
            )
        
        # Add note about negative scores
        negative_models = []
        for model_name, model_data in training_results['models'].items():
            score = model_data['test_scores'][primary_metric]
            if score < 0:
                negative_models.append(f"{model_name} ({score:.4f})")
        
        if negative_models:
            fig.add_annotation(
                text=f"*Models with negative {primary_metric}: {', '.join(negative_models)}",
                xref="paper", yref="paper",
                x=0, y=-0.15,
                showarrow=False,
                font=dict(size=10, color="red")
            )
        
        # Update layout
        layout = self.get_layout(
            title=f"Model Performance Comparison ({primary_metric})",
            xaxis_title="Model",
            yaxis_title=primary_metric.replace('_', ' ').title()
        )
        
        fig.update_layout(layout)
        fig.update_xaxes(tickangle=-45)
        
        return fig
    
    def create_confusion_matrices(self, training_results: Dict[str, Any]) -> go.Figure:
        """Create confusion matrices for classification models"""
        
        # Get top 4 models
        sorted_models = sorted(
            training_results['models'].items(),
            key=lambda x: x[1]['test_scores']['accuracy'],
            reverse=True
        )[:4]
        
        n_models = len(sorted_models)
        rows = 2
        cols = 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[model[0] for model in sorted_models],
            vertical_spacing=min(0.15, 0.9 / max(rows, 1)),
            horizontal_spacing=0.1
        )
        
        for idx, (model_name, model_data) in enumerate(sorted_models):
            row = idx // cols + 1
            col = idx % cols + 1
            
            # Get confusion matrix
            y_true = training_results.get('y_test', [])
            y_pred = model_data['predictions']
            
            if len(y_true) > 0:
                cm = confusion_matrix(y_true, y_pred)
                
                # Create heatmap
                heatmap = go.Heatmap(
                    z=cm,
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    colorscale='Blues',
                    showscale=idx == 0
                )
                
                fig.add_trace(heatmap, row=row, col=col)
                
                # Update axes
                fig.update_xaxes(title_text="Predicted", row=row, col=col)
                fig.update_yaxes(title_text="Actual", row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title_text="Confusion Matrices",
            height=800,
            template=self.template
        )
        
        return fig
    
    def create_model_comparison_heatmap(self, training_results):
        """Create a heatmap comparing model performance across metrics (column-normalized)"""
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np

        # Debug logging
        self.logger.debug(f"Creating heatmap with results type: {type(training_results)}")

        # Validate input
        if not isinstance(training_results, dict) or 'models' not in training_results:
            self.logger.warning("Invalid training_results format for heatmap")
            fig = go.Figure()
            fig.add_annotation(
                text="No model data available for heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        models_data = training_results.get('models', {})

        if not models_data:
            self.logger.warning("No models in training_results")
            fig = go.Figure()
            fig.add_annotation(
                text="No models have been trained yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        # Collect all metrics and models
        all_metrics = set()
        model_names = []

        for model_name, model_data in models_data.items():
            model_names.append(model_name)
            if 'test_scores' in model_data and model_data['test_scores']:
                all_metrics.update(model_data['test_scores'].keys())
                self.logger.debug(f"{model_name} metrics: {list(model_data['test_scores'].keys())}")

        if not all_metrics:
            self.logger.warning("No metrics found in any model")
            fig = go.Figure()
            fig.add_annotation(
                text="No performance metrics available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        # Sort for consistent ordering
        metrics_list = sorted(list(all_metrics))
        model_names_sorted = sorted(model_names)

        # Create matrix for raw values and text labels
        matrix = []
        text_matrix = []

        for model_name in model_names_sorted:
            row = []
            text_row = []
            model_data = models_data.get(model_name, {})
            test_scores = model_data.get('test_scores', {})

            for metric in metrics_list:
                value = test_scores.get(metric, np.nan)
                row.append(value if not np.isnan(value) else 0)
                text_row.append(f'{value:.4f}' if not np.isnan(value) else 'N/A')

            matrix.append(row)
            text_matrix.append(text_row)

        # Define metric directions (before normalization)
        higher_is_better = {
            'accuracy', 'precision', 'recall', 'f1_score', 'r2_score', 'r2',
            'explained_variance', 'roc_auc', 'auc', 'average_precision'
        }

        lower_is_better = {
            'mse', 'rmse', 'mae', 'mape', 'mean_squared_error', 
            'mean_absolute_error', 'median_absolute_error', 'max_error',
            'log_loss', 'mean_squared_log_error'
        }

        # COLUMN-WISE NORMALIZATION with direction awareness
        matrix_np = np.array(matrix, dtype=float)
        matrix_normalized = np.zeros_like(matrix_np)

        for j, metric in enumerate(metrics_list):
            col = matrix_np[:, j]
            valid = ~np.isnan(col)
            
            if not np.any(valid):
                matrix_normalized[:, j] = 0.5
                continue
                
            cmin = np.nanmin(col)
            cmax = np.nanmax(col)
            
            if np.allclose(cmax, cmin):
                # No variation; set to mid color
                matrix_normalized[:, j] = 0.5
            else:
                # Check metric direction
                metric_lower = metric.lower()
                
                # Normalize based on direction
                if any(m in metric_lower for m in lower_is_better):
                    # For "lower is better" metrics, invert the normalization
                    # Lower values -> higher normalized scores (darker colors)
                    matrix_normalized[valid, j] = 1 - (col[valid] - cmin) / (cmax - cmin)
                else:
                    # For "higher is better" metrics
                    # Higher values -> higher normalized scores (darker colors)
                    matrix_normalized[valid, j] = (col[valid] - cmin) / (cmax - cmin)
        
        matrix_normalized = np.nan_to_num(matrix_normalized, nan=0.5)

        # Create labels with arrows
        x_labels = []
        for metric in metrics_list:
            metric_lower = metric.lower()
            label = metric.replace('_', ' ').title()
            
            if any(m in metric_lower for m in higher_is_better):
                label += ' ↑'  # Up arrow for "higher is better"
            elif any(m in metric_lower for m in lower_is_better):
                label += ' ↓'  # Down arrow for "lower is better"
            
            x_labels.append(label)  # FIXED: This line was indented too far
        
        # Create heatmap with direction-aware normalized values
        fig = go.Figure(data=go.Heatmap(
            z=matrix_normalized.tolist(),  # Use direction-aware normalized matrix
            x=x_labels,  # Use the labels with arrows
            y=model_names_sorted,
            colorscale='YlGnBu',  # Softer palette
            text=text_matrix,
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))

        fig.update_layout(
            title='Model Performance Heatmap (Column-wise Normalized)',
            xaxis_title='Metric',
            yaxis_title='Model',
            height=max(400, 200 + len(model_names_sorted) * 40),
            width=max(600, 400 + len(metrics_list) * 100)
        )

        self.logger.info(f"Created heatmap with {len(model_names_sorted)} models and {len(metrics_list)} metrics")

        return fig

    def create_pairwise_comparison(self, training_results: Dict[str, Any], model1: str, model2: str) -> go.Figure:
        """Create pairwise model comparison"""
        
        # Check if models exist
        if (not training_results.get('models') or 
            model1 not in training_results['models'] or 
            model2 not in training_results['models']):
            fig = go.Figure()
            fig.add_annotation(
                text="Selected models not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        model1_data = training_results['models'][model1]
        model2_data = training_results['models'][model2]
        
        # Get metrics and handle negative values
        metrics = list(model1_data['test_scores'].keys())
        scores1 = []
        scores2 = []
        display_metrics = []
        
        for m in metrics:
            s1 = model1_data['test_scores'][m]
            s2 = model2_data['test_scores'][m]
            
            # Skip metrics where either model has negative value
            if s1 >= 0 and s2 >= 0:
                scores1.append(s1)
                scores2.append(s2)
                display_metrics.append(m)
        
        if not display_metrics:
            fig = go.Figure()
            fig.add_annotation(
                text="No positive metrics to compare",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatterpolar(
            r=scores1,
            theta=display_metrics,
            fill='toself',
            name=model1,
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=scores2,
            theta=display_metrics,
            fill='toself',
            name=model2,
            line_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f"Model Comparison: {model1} vs {model2}",
            template=self.template
        )
        
        return fig
    
    def create_aggregated_feature_importance(self, training_results: Dict[str, Any]) -> go.Figure:
        """Create aggregated feature importance across all models in a single view"""
        
        # Collect feature importance from all models
        all_models_importance = {}
        
        for model_name, model_data in training_results['models'].items():
            if 'feature_importance' in model_data and model_data['feature_importance'].get('importances'):
                importances = model_data['feature_importance']['importances']
                all_models_importance[model_name] = importances
        
        if not all_models_importance:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Calculate number of subplots needed
        n_models = len(all_models_importance)
        n_cols = 3  # 3 columns
        n_rows = (n_models + n_cols - 1) // n_cols
        
        # Create subplots with shared x-axis
        subplot_titles = list(all_models_importance.keys())
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=min(0.15, 0.9 / max(n_rows, 1)),
            horizontal_spacing=0.1,
            shared_xaxes=True  # Share x-axis across all subplots
        )
        
        # Debug: Check feature names format
        logger.debug(f"Number of models with feature importance: {n_models}")
        
        # Add each model's feature importance
        for idx, (model_name, importances) in enumerate(all_models_importance.items()):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            # Debug: Check the structure of importances
            if idx == 0:  # Only debug first model
                logger.debug(f"Model: {model_name}")
                logger.debug(f"Number of features: {len(importances)}")
                first_5_features = list(importances.keys())[:5]
                logger.debug(f"First 5 feature names: {first_5_features}")
            
            # Sort features by importance
            sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            
            # Extract feature names and values - ensure no duplication
            features = []
            values = []
            seen_features = set()
            
            for feat_name, importance_value in sorted_features:
                # Clean the feature name if needed
                clean_feat_name = str(feat_name).strip()
                
                # Check if this is a duplicated feature name
                if clean_feat_name not in seen_features:
                    features.append(clean_feat_name)
                    values.append(importance_value)
                    seen_features.add(clean_feat_name)
                else:
                    logger.warning(f"Duplicate feature name found: {clean_feat_name}")
            
            # Normalize values to [-1, 1] range while preserving sign
            max_abs_value = max(abs(v) for v in values) if values else 1
            if max_abs_value > 0:
                normalized_values = [v / max_abs_value for v in values]
            else:
                normalized_values = values
            
            # Color based on positive/negative
            colors = ['green' if v >= 0 else 'red' for v in normalized_values]
            
            # Create hover text with actual values
            hover_texts = [f'{feat}: {val:.4f}' for feat, val in zip(features, values)]
            
            fig.add_trace(
                go.Bar(
                    x=normalized_values,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    showlegend=False,
                    hovertext=hover_texts,
                    hovertemplate='%{hovertext}<extra></extra>',
                    name=model_name,
                    text=None,  # Explicitly set to None to avoid any text display
                    textposition=None  # Explicitly set to None
                ),
                row=row, col=col
            )
            
            # Update y-axis to ensure proper display
            fig.update_yaxes(
                tickfont=dict(size=9), 
                row=row, col=col,
                automargin=True,  # Allow more space for labels
                showticklabels=True,
                tickmode='array',
                tickvals=list(range(len(features))),
                ticktext=features  # Explicitly set the tick text
            )
        
        # Update x-axes to show normalized scale
        fig.update_xaxes(range=[-1.1, 1.1], title_text="Normalized Importance")
        
        # Update layout
        fig.update_layout(
            title="Feature Importance Across All Models (Normalized)",
            height=max(600, n_rows * 300),
            template=self.template,
            showlegend=False,
            margin=dict(l=150, r=50, t=100, b=50)  # Increase margins for better label display
        )
        
        # Add annotation explaining normalization
        fig.add_annotation(
            text="Note: Importance values are normalized to [-1, 1] within each model. Hover for actual values.",
            xref="paper", yref="paper",
            x=0.5, y=-0.02,
            showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        return fig
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, Any],
                                     model_name: str) -> go.Figure:
        """Create feature importance plot for single model"""
        
        if not feature_importance or not feature_importance.get('importances'):
            fig = go.Figure()
            fig.add_annotation(
                text=f"No feature importance available for {model_name}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get importances
        importances = feature_importance['importances']
        
        # Sort by importance
        sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        # Color based on positive/negative
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.4f}' for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Feature Importance - {model_name}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(features) * 25),
            template=self.template,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_cv_scores_plot(self, training_results: Dict[str, Any]) -> go.Figure:
        """Create cross-validation scores plot"""
        
        models = []
        means = []
        stds = []
        
        for model_name, model_data in training_results['models'].items():
            if 'cv_scores' in model_data and model_data['cv_scores']:
                # Skip models with negative CV scores
                if model_data['cv_scores']['mean'] >= 0:
                    models.append(model_name)
                    means.append(model_data['cv_scores']['mean'])
                    stds.append(model_data['cv_scores']['std'])
        
        if not models:
            fig = go.Figure()
            fig.add_annotation(
                text="No positive cross-validation data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Sort by mean score
        sorted_indices = np.argsort(means)[::-1]
        models = [models[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        
        fig = go.Figure()
        
        # Add error bars
        fig.add_trace(go.Scatter(
            x=models,
            y=means,
            error_y=dict(
                type='data',
                array=stds,
                visible=True
            ),
            mode='markers+lines',
            marker=dict(size=10, color='blue'),
            line=dict(color='blue', width=2),
            name='CV Score'
        ))
        
        # Add individual fold scores if available
        for i, model_name in enumerate(models):
            model_data = training_results['models'][model_name]
            if 'cv_scores' in model_data and 'scores' in model_data['cv_scores']:
                fold_scores = model_data['cv_scores']['scores']
                # Filter out negative scores
                fold_scores = [s for s in fold_scores if s >= 0]
                if fold_scores:
                    fig.add_trace(go.Scatter(
                        x=[model_name] * len(fold_scores),
                        y=fold_scores,
                        mode='markers',
                        marker=dict(size=6, color='lightblue', symbol='circle-open'),
                        name='Fold Scores' if i == 0 else None,
                        showlegend=i == 0,
                        hovertemplate=f'{model_name}<br>Fold Score: %{{y:.4f}}<extra></extra>'
                    ))
        
        fig.update_layout(
            title="Cross-Validation Scores by Model",
            xaxis_title="Model",
            yaxis_title="Score",
            template=self.template,
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_learning_curve(self, learning_curve_data: Dict[str, Any]) -> go.Figure:
        """Create learning curve visualization"""
        
        if not learning_curve_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No learning curve data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        train_sizes = learning_curve_data['train_sizes']
        train_mean = learning_curve_data['train_scores_mean']
        train_std = learning_curve_data['train_scores_std']
        val_mean = learning_curve_data['val_scores_mean']
        val_std = learning_curve_data['val_scores_std']
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Training confidence interval
        fig.add_trace(go.Scatter(
            x=train_sizes + train_sizes[::-1],
            y=[m + s for m, s in zip(train_mean, train_std)] + 
              [m - s for m, s in zip(train_mean[::-1], train_std[::-1])],
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        # Validation confidence interval
        fig.add_trace(go.Scatter(
            x=train_sizes + train_sizes[::-1],
            y=[m + s for m, s in zip(val_mean, val_std)] + 
              [m - s for m, s in zip(val_mean[::-1], val_std[::-1])],
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title="Learning Curve",
            xaxis_title="Training Set Size",
            yaxis_title="Score",
            template=self.template,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_stability_analysis(self, training_results: Dict[str, Any]) -> go.Figure:
        """Create model stability analysis across CV folds"""
        
        models_with_cv = []
        cv_scores_data = []
        
        for model_name, model_data in training_results['models'].items():
            if 'cv_scores' in model_data and 'scores' in model_data['cv_scores']:
                # Filter out models with negative scores
                scores = [s for s in model_data['cv_scores']['scores'] if s >= 0]
                if scores:
                    models_with_cv.append(model_name)
                    cv_scores_data.append(scores)
        
        if not models_with_cv:
            fig = go.Figure()
            fig.add_annotation(
                text="No positive cross-validation fold data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = go.Figure()
        
        # Create box plots
        for i, (model, scores) in enumerate(zip(models_with_cv, cv_scores_data)):
            fig.add_trace(go.Box(
                y=scores,
                name=model,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker_color=self.categorical_colors[i % len(self.categorical_colors)]
            ))
        
        fig.update_layout(
            title="Model Stability Analysis (CV Fold Variance)",
            yaxis_title="Score",
            template=self.template,
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_error_analysis(self, model_data: Dict[str, Any],
                            processed_data: Dict[str, Any]) -> go.Figure:
        """Create error analysis for regression models"""
        
        y_true = processed_data['y_test']
        y_pred = model_data['predictions']
        
        errors = y_true - y_pred
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction vs Actual', 'Residual Plot',
                          'Error Distribution', 'Q-Q Plot'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Prediction vs Actual
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers',
                      marker=dict(size=5, opacity=0.6),
                      name='Predictions'),
            row=1, col=1
        )
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', line=dict(color='red', dash='dash'),
                      name='Perfect Prediction'),
            row=1, col=1
        )
        
        # 2. Residual Plot
        fig.add_trace(
            go.Scatter(x=y_pred, y=errors, mode='markers',
                      marker=dict(size=5, opacity=0.6),
                      name='Residuals'),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Error Distribution
        fig.add_trace(
            go.Histogram(x=errors, nbinsx=30, name='Error Distribution'),
            row=2, col=1
        )
        
        # 4. Q-Q Plot
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(errors)))
        sample_quantiles = np.sort(errors)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                      mode='markers', marker=dict(size=5),
                      name='Q-Q Plot'),
            row=2, col=2
        )
        
        # Add reference line
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles,
                      y=theoretical_quantiles * np.std(errors) + np.mean(errors),
                      mode='lines', line=dict(color='red', dash='dash'),
                      name='Normal'),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Actual", row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_xaxes(title_text="Predicted", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=1, col=2)
        fig.update_xaxes(title_text="Error", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        
        fig.update_layout(
            title_text="Regression Error Analysis",
            height=800,
            template=self.template,
            showlegend=False
        )
        
        return fig
    
    def create_feature_prediction_correlation(self, training_results: Dict[str, Any],
                                            processed_data: Dict[str, Any]) -> go.Figure:
        """Create correlation between features and predictions"""
        
        # Get best model predictions
        best_model_data = training_results['best_model_data']
        predictions = best_model_data['predictions']
        
        # Get test features
        X_test = processed_data['X_test']
        
        # Calculate correlations
        correlations = {}
        for col in X_test.columns[:20]:  # Limit to top 20 features
            corr = np.corrcoef(X_test[col], predictions)[0, 1]
            correlations[col] = corr
        
        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        features = [c[0] for c in sorted_corrs]
        corr_values = [c[1] for c in sorted_corrs]
        colors = ['green' if c >= 0 else 'red' for c in corr_values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=corr_values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f'{c:.3f}' for c in corr_values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Feature-Prediction Correlations",
            xaxis_title="Correlation with Predictions",
            yaxis_title="Feature",
            height=max(400, len(features) * 25),
            template=self.template,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filename: str,
                   output_dir: Optional[Path] = None) -> Path:
        """Save figure to file
        
        Args:
            fig: Plotly figure
            filename: Output filename
            output_dir: Output directory (default: aquavista_results/reports)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path("aquavista_results/reports")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        
        # Determine format from extension
        if filename.endswith('.html'):
            fig.write_html(filepath)
        elif filename.endswith('.png'):
            fig.write_image(filepath)
        elif filename.endswith('.pdf'):
            fig.write_image(filepath)
        elif filename.endswith('.svg'):
            fig.write_image(filepath)
        else:
            # Default to HTML
            filepath = filepath.with_suffix('.html')
            fig.write_html(filepath)
        
        logger.info(f"Figure saved to {filepath}")
        
        return filepath
    # --- New helper builders ---

    def _bar_from_mapping(self, mapping: dict, title: str, xlab: str, ylab: str, top: int = 20):
        import plotly.graph_objects as go
        if not mapping:
            return go.Figure()
        items = sorted(mapping.items(), key=lambda kv: (kv[1] is None, 0 if kv[1] is None else -kv[1]))[:top]
        labels = [k for k, _ in items]
        vals = [0.0 if v is None else float(v) for _, v in items]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation='h',
                            text=[f"{v:.3g}" for v in vals], textposition='outside'))
        fig.update_layout(**self.get_layout(title=title, xaxis_title=xlab, yaxis_title=ylab,
                                            height=max(420, 24*len(labels))))
        fig.update_yaxes(categoryorder='total ascending')
        return fig

    def _create_vif_plot(self, stats: dict):
        mc = stats.get('multicollinearity') or {}
        vif = mc.get('vif_scores') or mc.get('vif') or {}
        if isinstance(vif, list):
            vif = {row['feature']: row.get('vif') for row in vif
                if isinstance(row, dict) and 'feature' in row}

        fig = self._bar_from_mapping(vif, "VIF by Feature", "VIF", "Feature")

        if vif:  # add guides only if there is something to show
            fig.add_vline(
                x=5, line_dash="dot", line_color="orange",
                annotation_text="VIF ≈ 5", annotation_position="top right"
            )
            fig.add_vline(
                x=10, line_dash="dash", line_color="red",
                annotation_text="VIF ≈ 10", annotation_position="top right"
            )
        return fig


    def _create_mi_plot(self, stats: dict):
        fr = stats.get('feature_relationships') or {}
        mi = fr.get('mutual_information') or fr.get('mutual_info') or {}
        return self._bar_from_mapping(mi, "Mutual Information (feature → target)", "MI score", "Feature")

    def _create_anova_plot(self, stats: dict):
        import numpy as np
        fr = stats.get('feature_relationships') or {}
        pvals = fr.get('anova_pvalues') or fr.get('anova') or {}
        neglog = {k: (0.0 if v in [None, 0] else -np.log10(float(v))) for k, v in pvals.items()}
        return self._bar_from_mapping(neglog, "ANOVA significance (−log₁₀ p)", "−log₁₀(p)", "Feature")

    def _create_chi2_plot(self, stats: dict):
        import numpy as np
        fr = stats.get('feature_relationships') or {}
        pvals = fr.get('chi2_pvalues') or fr.get('chi2') or {}
        neglog = {k: (0.0 if v in [None, 0] else -np.log10(float(v))) for k, v in pvals.items()}
        return self._bar_from_mapping(neglog, "Chi-square significance (−log₁₀ p)", "−log₁₀(p)", "Feature")


    def _create_hetero_plot(self, stats: dict):
        stests = stats.get('statistical_tests') or {}
        het = stests.get('heteroscedasticity') or {}
        mapping = {}
        for k, v in het.items():
            mapping[k] = (v.get('levene_p') if isinstance(v, dict) else v)
        fig = self._bar_from_mapping(mapping, "Heteroscedasticity (Levene p-values)", "p-value", "Feature")
        fig.add_vline(x=0.05, line_dash="dash", line_color="red", annotation_text="p=0.05")
        return fig

    def _create_outlier_plot(self, stats: dict):
        out = stats.get('outlier_analysis') or {}
        counts = {}
        for feat, d in out.items():
            if isinstance(d, dict) and 'iqr_method' in d:
                counts[feat] = d['iqr_method'].get('n_outliers', 0)
        return self._bar_from_mapping(counts, "Outliers per Feature (IQR method)", "# outliers", "Feature")

    def _create_ts_table(self, stats: dict):
        import plotly.graph_objects as go
        ts = stats.get('time_series_check') or {}
        if not ts:
            return go.Figure()

        cols = ["Feature", "ADF p", "KPSS p", "Seasonal Period", "Top ACF Lags (1-10)"]
        rows = []
        for feat, r in ts.items():
            top = r.get('acf_top_lags', []) or []
            top_txt = ", ".join([f"{t['lag']}d:{t['acf']:.2f}" for t in top]) if top else "—"
            rows.append([
                feat,
                f"{r.get('adf_pvalue', float('nan')):.3g}",
                f"{r.get('kpss_pvalue', float('nan')):.3g}",
                str(r.get('seasonal_period', '—')),
                top_txt
            ])

        fig = go.Figure(data=[go.Table(
            header=dict(values=cols, fill_color='lightgray', align='left'),
            cells=dict(values=list(zip(*rows)), align='left')
        )])

        # Dynamic height: 140 base + 24 px per row (cap at 900)
        table_height = min(900, 140 + 24 * max(1, len(rows)))
        fig.update_layout(title="Time-Series Diagnostics (Daily)", height=table_height)
        return fig


    def _create_acf_daily_plot(self, stats: dict):
        import plotly.graph_objects as go
        ts = stats.get('time_series_check') or {}
        if not ts:
            return go.Figure()
        feat = next((k for k, v in ts.items() if v.get('acf_by_lag')), None)
        if not feat:
            return go.Figure()
        acf_map = ts[feat]['acf_by_lag']
        lags = list(acf_map.keys())
        vals = [acf_map[l] for l in lags]
        fig = go.Figure(go.Bar(x=lags, y=vals, text=[f"{v:.2f}" for v in vals], textposition='outside'))
        fig.update_layout(**self.get_layout(title=f"Daily ACF (lags 1–10): {feat}",
                                            xaxis_title="Lag (days)", yaxis_title="ACF"))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        return fig

    def create_statistical_plots(self, statistical_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create plots for statistical analysis results"""
        
        plots = {}
        
        # 1. Distribution plots
        if 'distribution_analysis' in statistical_results:
            plots['distributions'] = self._create_distribution_plots(
                statistical_results['distribution_analysis']
            )
        
        # 2. Correlation heatmap
        if 'correlation_analysis' in statistical_results:
            plots['correlation_heatmap'] = self._create_correlation_heatmap(
                statistical_results['correlation_analysis']
            )
        
        # 3. PCA visualization
        if 'dimensionality_reduction' in statistical_results:
            plots['pca_analysis'] = self._create_pca_plots(
                statistical_results['dimensionality_reduction']
            )
        
        # 4. Clustering visualization
        if 'clustering_analysis' in statistical_results:
            plots['clustering'] = self._create_clustering_plots(
                statistical_results['clustering_analysis']
            )
        plots['vif'] = self._create_vif_plot(statistical_results)
        plots['mutual_info'] = self._create_mi_plot(statistical_results)
        plots['anova'] = self._create_anova_plot(statistical_results)
        plots['chi2'] = self._create_chi2_plot(statistical_results)
        plots['heteroscedasticity'] = self._create_hetero_plot(statistical_results)
        plots['outliers'] = self._create_outlier_plot(statistical_results)
        if statistical_results.get('time_series_check'):
            plots['ts_diagnostics'] = self._create_ts_table(statistical_results)
            plots['acf_daily'] = self._create_acf_daily_plot(statistical_results)

        return plots
    
    def _create_distribution_plots(self, distribution_data: Dict[str, Any]) -> go.Figure:
        """Create distribution analysis plots"""
        
        # Select top features by non-normality
        features = []
        normality_scores = []
        
        for feature, data in distribution_data.items():
            if 'normality' in data:
                features.append(feature)
                normality_scores.append(data['normality']['dagostino_p'])
        
        # Sort by p-value (lower = less normal)
        sorted_indices = np.argsort(normality_scores)[:10]
        selected_features = [features[i] for i in sorted_indices]
        
        fig = go.Figure()
        
        for i, feature in enumerate(selected_features):
            fig.add_trace(go.Bar(
                x=[feature],
                y=[normality_scores[sorted_indices[i]]],
                name=feature,
                marker_color=self.categorical_colors[i % len(self.categorical_colors)]
            ))
        
        # Add significance line
        fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                     annotation_text="p=0.05 (normality threshold)")
        
        fig.update_layout(
            title="Distribution Normality Test (D'Agostino)",
            xaxis_title="Feature",
            yaxis_title="p-value",
            yaxis_type="log",
            template=self.template,
            showlegend=False,
            height=500
        )
        
        return fig
    
    def _create_correlation_heatmap(self, correlation_data: Dict[str, Any]) -> go.Figure:
        """Create correlation heatmap"""
        
        if 'pearson' not in correlation_data:
            return go.Figure()
        
        # Convert correlation dict to matrix
        corr_dict = correlation_data['pearson']
        features = list(corr_dict.keys())
        
        # Create correlation matrix
        n = len(features)
        corr_matrix = np.zeros((n, n))
        
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if feat2 in corr_dict[feat1]:
                    corr_matrix[i, j] = corr_dict[feat1][feat2]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=features,
            y=features,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{x} - %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            template=self.template,
            height=600,
            width=800
        )
        
        return fig

    def create_scaler_comparison_chart(self, scaler_results):
        """Create a chart comparing scaler performance across models"""
        import plotly.graph_objects as go
        import pandas as pd
        
        # Prepare data
        models = []
        scalers = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None']
        scaler_scores = {s: [] for s in scalers}
        
        for model_name, results in scaler_results.items():
            models.append(model_name)
            scores = results.get('scores', {})
            for scaler in scalers:
                scaler_scores[scaler].append(scores.get(scaler, 0))
        
        # Create figure
        fig = go.Figure()
        
        # Color palette
        colors = {
            'StandardScaler': '#1f77b4',
            'MinMaxScaler': '#ff7f0e', 
            'RobustScaler': '#2ca02c',
            'None': '#d62728'
        }
        
        # Add bars for each scaler
        for scaler in scalers:
            scaler_label = scaler.replace('Scaler', '').replace('None', 'No Scaling')
            fig.add_trace(go.Bar(
                name=scaler_label,
                x=models,
                y=scaler_scores[scaler],
                marker_color=colors[scaler],
                text=[f'{s:.4f}' if s > 0 else 'Failed' for s in scaler_scores[scaler]],
                textposition='outside',
                textfont_size=10
            ))
        
        # Update layout
        fig.update_layout(
            title='Scaler Performance Comparison by Model',
            xaxis_title='Model',
            yaxis_title='Cross-Validation Score',
            barmode='group',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=500
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=-45)
        
        return fig

    def create_scaler_selection_matrix(self, scaler_results):
        """Create a matrix showing which scaler was selected for each model"""
        import plotly.graph_objects as go
        import numpy as np
        
        models = list(scaler_results.keys())
        scalers = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None']
        
        # Create matrix
        matrix = []
        text_matrix = []
        for model in models:
            row = []
            text_row = []
            best_scaler = scaler_results[model].get('best_scaler', '')
            scores = scaler_results[model].get('scores', {})
            
            for scaler in scalers:
                score = scores.get(scaler, 0)
                if scaler == best_scaler:
                    row.append(score)  # Use actual score for best
                    text_row.append(f'✓ {score:.4f}')
                else:
                    row.append(score * 0.7)  # Dimmed for non-best
                    text_row.append(f'{score:.4f}')
            matrix.append(row)
            text_matrix.append(text_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[s.replace('Scaler', '').replace('None', 'No Scaling') for s in scalers],
            y=models,
            colorscale='Viridis',
            text=text_matrix,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Scaler Selection Matrix (✓ = Selected)',
            xaxis_title='Scaler Type',
            yaxis_title='Model',
            template='plotly_white',
            height=max(400, len(models) * 30)
        )
        
        return fig
    
    def _create_pca_plots(self, pca_data: Dict[str, Any]) -> go.Figure:
        """Create PCA analysis plots"""
        
        if 'pca' not in pca_data or 'error' in pca_data['pca']:
            return go.Figure()
        
        pca_results = pca_data['pca']
        
        # Create variance explained plot
        var_exp = pca_results['explained_variance_ratio']
        cum_var = pca_results['cumulative_variance_ratio']
        
        fig = go.Figure()
        
        # Individual variance
        fig.add_trace(go.Bar(
            x=list(range(1, len(var_exp) + 1)),
            y=var_exp,
            name='Individual',
            marker_color='lightblue'
        ))
        
        # Cumulative variance
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cum_var) + 1)),
            y=cum_var,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        # Add 95% threshold line
        fig.add_hline(y=0.95, line_dash="dash", line_color="green",
                     annotation_text="95% variance explained")
        
        fig.update_layout(
            title="PCA Variance Explained",
            xaxis_title="Principal Component",
            yaxis_title="Variance Explained Ratio",
            template=self.template,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_clustering_plots(self, clustering_data: Dict[str, Any]) -> go.Figure:
        """Create clustering analysis plots"""
        
        if 'kmeans' not in clustering_data or 'error' in clustering_data['kmeans']:
            return go.Figure()
        
        kmeans_results = clustering_data['kmeans']
        scores = kmeans_results['scores']
        
        # Extract metrics
        k_values = [s['k'] for s in scores]
        silhouette = [s['silhouette'] for s in scores]
        inertia = [s['inertia'] for s in scores]
        
        # Normalize inertia for dual y-axis
        inertia_norm = np.array(inertia)
        inertia_norm = (inertia_norm - inertia_norm.min()) / (inertia_norm.max() - inertia_norm.min())
        
        fig = go.Figure()
        
        # Silhouette score
        fig.add_trace(go.Scatter(
            x=k_values,
            y=silhouette,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Normalized inertia
        fig.add_trace(go.Scatter(
            x=k_values,
            y=inertia_norm,
            mode='lines+markers',
            name='Inertia (normalized)',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Mark optimal k
        optimal_k = kmeans_results['optimal_k']
        optimal_idx = k_values.index(optimal_k)
        
        fig.add_trace(go.Scatter(
            x=[optimal_k],
            y=[silhouette[optimal_idx]],
            mode='markers',
            name=f'Optimal k={optimal_k}',
            marker=dict(size=15, color='green', symbol='star')
        ))
        
        fig.update_layout(
            title="K-Means Clustering Analysis",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Score",
            template=self.template,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    # ADD THESE METHODS TO YOUR VisualizationEngine CLASS IN visualization.py
    # Add them around line 1500, before the end of the class

    def create_comprehensive_ranking_chart(self, ranking_results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive model ranking visualization with multiple views"""
        
        if not ranking_results or 'rankings' not in ranking_results:
            fig = go.Figure()
            fig.add_annotation(text="No ranking data available", 
                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        rankings = ranking_results['rankings']
        
        # Create subplot for different perspectives
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Overall Rankings', 'Performance vs Interpretability', 
                        'Efficiency vs Stability', 'Top Model Score Breakdown'],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                [{'secondary_y': False}, {'type': 'domain'}]]
        )
        
        # Extract data for visualizations
        models = [r['model'] for r in rankings]
        overall_scores = [r['overall_score'] for r in rankings]
        perf_scores = [r['performance_score'] for r in rankings]
        interp_scores = [r['interpretability_score'] for r in rankings]
        efficiency_scores = [r['efficiency_score'] for r in rankings]
        stability_scores = [r['stability_score'] for r in rankings]
        
        # 1. Overall ranking bar chart with color gradient
        colors = ['gold' if i == 0 else 'silver' if i == 1 else 'brown' if i == 2 else 'lightblue' 
                for i in range(len(models))]
        
        fig.add_trace(
            go.Bar(
                x=models, y=overall_scores, 
                name='Overall Score',
                marker_color=colors, 
                text=[f'{s:.3f}' for s in overall_scores],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Overall Score: %{y:.3f}<extra></extra>'
            ), 
            row=1, col=1
        )
        
        # 2. Performance vs Interpretability scatter
        fig.add_trace(
            go.Scatter(
                x=interp_scores, y=perf_scores, 
                mode='markers+text', 
                text=models, 
                textposition='top center',
                marker=dict(
                    size=[score * 20 + 5 for score in overall_scores],  # Size based on overall score
                    color=overall_scores,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Overall Score")
                ),
                name='Models',
                hovertemplate='<b>%{text}</b><br>Performance: %{y:.3f}<br>Interpretability: %{x:.3f}<extra></extra>'
            ), 
            row=1, col=2
        )
        
        # 3. Efficiency vs Stability scatter
        fig.add_trace(
            go.Scatter(
                x=stability_scores, y=efficiency_scores,
                mode='markers+text', 
                text=models, 
                textposition='top center', 
                marker=dict(
                    size=[score * 20 + 5 for score in overall_scores],
                    color='green',
                    opacity=0.7
                ),
                name='Models',
                hovertemplate='<b>%{text}</b><br>Efficiency: %{y:.3f}<br>Stability: %{x:.3f}<extra></extra>'
            ), 
            row=2, col=1
        )
        
        # 4. Top model breakdown (pie chart)
        if rankings:
            top_model = rankings[0]
            breakdown_labels = ['Performance', 'Stability', 'Interpretability', 'Efficiency', 'Robustness']
            breakdown_values = [
                top_model['performance_score'], 
                top_model['stability_score'],
                top_model['interpretability_score'], 
                top_model['efficiency_score'],
                top_model['robustness_score']
            ]
            
            fig.add_trace(
                go.Pie(
                    labels=breakdown_labels, 
                    values=breakdown_values,
                    name=f"{top_model['model']} Breakdown",
                    hovertemplate='<b>%{label}</b><br>Score: %{value:.3f}<br>Percentage: %{percent}<extra></extra>'
                ), 
                row=2, col=2
            )
        
        # Update layout with improved styling
        fig.update_layout(
            height=800, 
            showlegend=False,
            title_text=f"Comprehensive Model Rankings - {ranking_results.get('use_case_priority', 'Balanced').title()} Priority",
            title_x=0.5,
            template='plotly_white'
        )
        
        # Update axis labels and styling
        fig.update_xaxes(title_text="Model", row=1, col=1, tickangle=45)
        fig.update_yaxes(title_text="Overall Score", row=1, col=1, range=[0, 1])
        
        fig.update_xaxes(title_text="Interpretability Score", row=1, col=2, range=[0, 1])
        fig.update_yaxes(title_text="Performance Score", row=1, col=2, range=[0, 1])
        
        fig.update_xaxes(title_text="Stability Score", row=2, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Efficiency Score", row=2, col=1, range=[0, 1])
        
        return fig

    def create_ranking_comparison_chart(self, ranking_results: Dict[str, Any]) -> go.Figure:
        """Create detailed comparison chart showing all criteria scores"""
        
        if not ranking_results or 'rankings' not in ranking_results:
            return go.Figure()
        
        rankings = ranking_results['rankings']
        
        # Prepare data for grouped bar chart
        models = [r['model'] for r in rankings]
        criteria = ['Performance', 'Stability', 'Interpretability', 'Efficiency', 'Robustness']
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, criterion in enumerate(criteria):
            scores = [r[f'{criterion.lower()}_score'] for r in rankings]
            
            fig.add_trace(go.Bar(
                name=criterion,
                x=models,
                y=scores,
                marker_color=colors[i],
                hovertemplate=f'<b>{criterion}</b><br>Model: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>'
            ))
        
        fig.update_layout(
            barmode='group',
            title='Detailed Model Comparison Across All Criteria',
            xaxis_title='Models',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            height=500
        )
        
        return fig

    def create_model_ranking_summary(self, ranking_results: Dict[str, Any]) -> go.Figure:
        """Create a summary visualization showing key insights from rankings"""
        
        if not ranking_results or 'rankings' not in ranking_results:
            return go.Figure()
        
        rankings = ranking_results['rankings']
        
        # Create summary metrics
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Overall Score Distribution', 'Performance Leaders', 'Interpretability Leaders'],
            specs=[[{'type': 'histogram'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 1. Score distribution histogram
        overall_scores = [r['overall_score'] for r in rankings]
        fig.add_trace(
            go.Histogram(
                x=overall_scores,
                nbinsx=10,
                name='Score Distribution',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. Top 3 performance leaders
        top_performers = sorted(rankings, key=lambda x: x['performance_score'], reverse=True)[:3]
        fig.add_trace(
            go.Bar(
                x=[model['model'] for model in top_performers],
                y=[model['performance_score'] for model in top_performers],
                name='Performance Leaders',
                marker_color='green',
                text=[f'{score:.3f}' for score in [model['performance_score'] for model in top_performers]],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. Top 3 interpretability leaders
        top_interpretable = sorted(rankings, key=lambda x: x['interpretability_score'], reverse=True)[:3]
        fig.add_trace(
            go.Bar(
                x=[model['model'] for model in top_interpretable],
                y=[model['interpretability_score'] for model in top_interpretable],
                name='Interpretability Leaders',
                marker_color='orange',
                text=[f'{score:.3f}' for score in [model['interpretability_score'] for model in top_interpretable]],
                textposition='outside'
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Model Ranking Summary Dashboard"
        )
        
        return fig
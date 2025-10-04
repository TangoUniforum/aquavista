"""
Statistical Analysis Module for AquaVista v6.0
=============================================
Comprehensive statistical analysis including PCA, correlation analysis,
hypothesis testing, and advanced statistical methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import scipy.stats as stats
from scipy.stats import (normaltest, shapiro, kstest, chi2_contingency,
                        f_oneway, kruskal, pearsonr, spearmanr, kendalltau)
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import (mutual_info_regression, mutual_info_classif,
                                     chi2, f_classif, f_regression)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
import logging

# Optional imports for advanced statistics
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss, acf

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Import visualization engine for plots
from modules.config import Config
from modules.visualization import VisualizationEngine

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class StatisticalAnalysis:
    """Comprehensive statistical analysis for machine learning data"""

    def _compute_pca_loadings(self, data: pd.DataFrame, n_components: int = 5):
        """Return (loadings_df, explained_variance_ratio_series) for numeric columns."""
        X = data.select_dtypes(include=[np.number]).copy()

        # Clean + impute
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))

        # Standardize for PCA
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # choose a safe number of components and fix randomness
        pca = PCA(n_components=min(n_components, Xs.shape[1]), random_state=0)
        pca.fit(Xs)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=X.columns,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        )

        evr = pd.Series(
            pca.explained_variance_ratio_,
            index=[f"PC{i+1}" for i in range(pca.n_components_)],
            name="ExplainedVarianceRatio"
        )
        return loadings, evr

    def _compute_pls_loadings(
        self,
        data: pd.DataFrame,
        target_column: str,
        n_components: int = 3,
    ):
        """
        Return (x_weights_df, evr_x_series) for PLS, or (None, None, reason) on skip.
        - x_weights_df: feature weights onto latent variables LV1..LVk
        - evr_x_series: approximate fraction of X variance captured by each LV
        """
        if not target_column or target_column not in data.columns:
            return None, None, "No valid target column for PLS."

        # --- Build X (numeric only) ---
        X = data.select_dtypes(include=[np.number]).copy()
        if X.empty:
            return None, None, "No numeric features available for PLS."

        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
        Xs = StandardScaler().fit_transform(X)

        # --- Build/encode y ---
        y_raw = data[target_column]
        if y_raw.isna().all():
            return None, None, "Target is all NaN."

        if y_raw.dtype.kind in "ifu":  # numeric target (regression)
            y = y_raw.values.reshape(-1, 1)
        else:
            # simple rule: if reasonably few classes, one-hot encode; else skip
            nunique = y_raw.nunique(dropna=True)
            if nunique <= 20:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                y = enc.fit_transform(y_raw.astype(str).to_numpy().reshape(-1, 1))
            else:
                return None, None, "Categorical target has too many classes; skipping PLS."

        # cap components so PLS is valid
        n_comp = max(1, min(n_components, Xs.shape[1], y.shape[1] if y.ndim > 1 else 1))

        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(Xs, y)

        # Feature weights onto latent variables
        x_weights = pd.DataFrame(
            pls.x_weights_,
            index=X.columns,
            columns=[f"LV{i+1}" for i in range(n_comp)],
        )

        # Approx EV in X by each LV (not a formal EVR, but a useful proxy)
        evr_x = np.var(pls.x_scores_, axis=0)
        denom = np.var(Xs, axis=0).sum()
        if denom > 0:
            evr_x = evr_x / denom
        evr_x = pd.Series(evr_x, index=x_weights.columns, name="ExplainedVarianceX")

        return x_weights, evr_x, None

    def __init__(self, config: Config):
        self.config = config
        self.viz_engine = VisualizationEngine(config)
        self.results = {}
        
    def run_complete_analysis(
        self,
        data: pd.DataFrame,
        target_column: str = None,
        task_type: str = None,
        n_components: int = 3,          
    ) -> Dict[str, Any]:
        """Run comprehensive statistical analysis on the dataset"""
        
        logger.info("Starting comprehensive statistical analysis")
        
        analysis_results = {
            'basic_statistics': self.calculate_basic_statistics(data),
            'distributions': self.analyze_distributions(data),
            'correlation_analysis': self.analyze_correlations(data, target_column),
            'multicollinearity': self.check_multicollinearity(data),
            'feature_relationships': self.analyze_feature_relationships(data, target_column, task_type),
            'dimensionality_reduction': self.perform_dimensionality_reduction(data),
            'clustering_analysis': self.perform_clustering_analysis(data),
            'statistical_tests': self.perform_statistical_tests(data, target_column, task_type),
            'outliers': self.analyze_outliers(data),
            'time_series': self.check_time_series_patterns(data) if self._has_datetime_index(data) else None,
        }

        # --- Step 1: PCA components (loadings) & explained variance ratio ---
        try:
            pca_loadings, pca_evr = self._compute_pca_loadings(data, n_components=5)
            analysis_results["pca_components"] = pca_loadings
            analysis_results["pca_evr"] = pca_evr
        except Exception as e:
            logger.warning(f"PCA loadings computation failed: {e}")

        # --- Step 2: PLS components (supervised) ---
        try:
            pls_w, pls_evr, _ = self._compute_pls_loadings(
                data, target_column, n_components=n_components
            )
            if pls_w is not None:
                analysis_results["pls_x_weights"] = pls_w
                analysis_results["pls_evr_x"] = pls_evr

        except Exception as e:
            logger.warning(f"PLS computation failed: {e}")

        # Store results
        self.results = analysis_results
        
        # Generate insights
        analysis_results['insights'] = self.generate_statistical_insights(analysis_results)
        
        logger.info("Statistical analysis complete")
        
        return analysis_results
    
    def calculate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical measures"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        categorical_data = data.select_dtypes(include=['object', 'category'])
        
        basic_stats = {
            'numeric': {
                'summary': numeric_data.describe().to_dict(),
                'skewness': numeric_data.skew().to_dict(),
                'kurtosis': numeric_data.kurtosis().to_dict(),
                'missing_values': numeric_data.isnull().sum().to_dict(),
                'unique_values': numeric_data.nunique().to_dict(),
                'zero_variance': (numeric_data.var() == 0).to_dict()
            },
            'categorical': {
                'unique_counts': {col: categorical_data[col].nunique() 
                                for col in categorical_data.columns},
                'mode': {col: categorical_data[col].mode()[0] if not categorical_data[col].mode().empty else None
                        for col in categorical_data.columns},
                'missing_values': categorical_data.isnull().sum().to_dict()
            },
            'dataset': {
                'n_samples': len(data),
                'n_features': len(data.columns),
                'n_numeric': len(numeric_data.columns),
                'n_categorical': len(categorical_data.columns),
                'memory_usage': data.memory_usage(deep=True).sum() / 1024**2  # MB
            }
        }
        
        return basic_stats
    
    def analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature distributions"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        distribution_results = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            
            if len(col_data) < 20:  # Skip if too few samples
                continue
            
            # Normality tests
            try:
                _, shapiro_p = shapiro(col_data) if len(col_data) < 5000 else (np.nan, np.nan)
                _, dagostino_p = normaltest(col_data)
                
                # Fit normal distribution
                mu, sigma = stats.norm.fit(col_data)
                
                # Kolmogorov-Smirnov test
                _, ks_p = kstest(col_data, 'norm', args=(mu, sigma))
                
                distribution_results[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'median': float(col_data.median()),
                    'mad': float(stats.median_abs_deviation(col_data)),
                    'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                    'range': float(col_data.max() - col_data.min()),
                    'cv': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else np.nan,
                    'normality': {
                        'shapiro_p': float(shapiro_p) if not np.isnan(shapiro_p) else None,
                        'dagostino_p': float(dagostino_p),
                        'ks_p': float(ks_p),
                        'is_normal': dagostino_p > 0.05
                    },
                    'distribution_type': self._identify_distribution(col_data)
                }
            except Exception as e:
                logger.warning(f"Could not analyze distribution for {col}: {e}")
                
        return distribution_results
    
    def analyze_correlations(self, data: pd.DataFrame, 
                           target_column: str = None) -> Dict[str, Any]:
        """Analyze correlations between features"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        correlation_results = {
            'pearson': numeric_data.corr(method='pearson').to_dict(),
            'spearman': numeric_data.corr(method='spearman').to_dict(),
            'kendall': numeric_data.corr(method='kendall').to_dict() if len(numeric_data) < 1000 else None
        }
        
        # Find highly correlated pairs
        corr_matrix = numeric_data.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_pairs = []
        for i in range(len(upper_tri.columns)):
            for j in range(i):
                if abs(upper_tri.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'feature1': upper_tri.columns[i],
                        'feature2': upper_tri.index[j],
                        'correlation': float(upper_tri.iloc[i, j])
                    })
        
        correlation_results['high_correlations'] = high_corr_pairs
        
        # Target correlations
        if target_column and target_column in numeric_data.columns:
            target_corr = {}
            target_data = numeric_data[target_column]
            
            for col in numeric_data.columns:
                if col != target_column:
                    pearson_corr, pearson_p = pearsonr(numeric_data[col], target_data)
                    spearman_corr, spearman_p = spearmanr(numeric_data[col], target_data)
                    
                    target_corr[col] = {
                        'pearson': float(pearson_corr),
                        'pearson_p': float(pearson_p),
                        'spearman': float(spearman_corr),
                        'spearman_p': float(spearman_p),
                        'abs_correlation': float(abs(pearson_corr))
                    }
            
            # Sort by absolute correlation
            correlation_results['target_correlations'] = dict(
                sorted(target_corr.items(), 
                      key=lambda x: x[1]['abs_correlation'], 
                      reverse=True)
            )
        
        return correlation_results
    
    def check_multicollinearity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for multicollinearity using VIF"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Remove columns with zero variance
        numeric_data = numeric_data.loc[:, numeric_data.var() > 0]
        
        multicollinearity_results = {
            'vif_scores': {},
            'high_vif_features': [],
            'condition_number': None
        }
        
        if STATSMODELS_AVAILABLE and len(numeric_data.columns) > 1:
            try:
                # Calculate VIF for each feature
                for i, col in enumerate(numeric_data.columns):
                    if len(numeric_data) > len(numeric_data.columns):
                        vif = variance_inflation_factor(numeric_data.values, i)
                        multicollinearity_results['vif_scores'][col] = float(vif) if not np.isinf(vif) else None
                        
                        if vif > 10:  # Common threshold
                            multicollinearity_results['high_vif_features'].append({
                                'feature': col,
                                'vif': float(vif)
                            })
                
                # Calculate condition number
                X = numeric_data.values
                if len(X) > 0:
                    cond_number = np.linalg.cond(X.T @ X)
                    multicollinearity_results['condition_number'] = float(cond_number)
                    
            except Exception as e:
                logger.warning(f"Could not calculate VIF: {e}")
        
        return multicollinearity_results
    
    def analyze_feature_relationships(self, data: pd.DataFrame, 
                                    target_column: str = None,
                                    task_type: str = None) -> Dict[str, Any]:
        """Analyze relationships between features and target"""
        
        if not target_column or target_column not in data.columns:
            return {}
        
        feature_relationships = {
            'mutual_information': {},
            'statistical_tests': {},
            'effect_sizes': {}
        }
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Separate numeric and categorical features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Mutual information
        if len(numeric_features) > 0:
            if task_type == 'regression' or (task_type is None and y.dtype in ['float64', 'int64']):
                mi_scores = mutual_info_regression(X[numeric_features], y)
            else:
                mi_scores = mutual_info_classif(X[numeric_features], y)
            
            for feat, score in zip(numeric_features, mi_scores):
                feature_relationships['mutual_information'][feat] = float(score)
        
        # Statistical tests based on task type
        if task_type == 'classification' or (task_type is None and y.nunique() < 20):
            # For classification
            for feat in numeric_features:
                # ANOVA for numeric features
                groups = [group[feat].values for name, group in data.groupby(target_column)]
                if len(groups) >= 2:
                    f_stat, p_value = f_oneway(*groups)
                    feature_relationships['statistical_tests'][feat] = {
                        'test': 'ANOVA',
                        'statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
            
            # Chi-square for categorical features
            for feat in categorical_features:
                contingency_table = pd.crosstab(X[feat], y)
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                
                feature_relationships['statistical_tests'][feat] = {
                    'test': 'Chi-square',
                    'statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'dof': int(dof),
                    'significant': p_value < 0.05
                }
        
        return feature_relationships
    
    def perform_dimensionality_reduction(self, data: pd.DataFrame,
                                       n_components: int = None) -> Dict[str, Any]:
        """Perform various dimensionality reduction techniques"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Remove columns with zero variance
        numeric_data = numeric_data.loc[:, numeric_data.var() > 0]
        
        if len(numeric_data.columns) < 3:
            return {'error': 'Insufficient numeric features for dimensionality reduction'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        if n_components is None:
            n_components = min(10, len(numeric_data.columns) - 1)
        
        dim_reduction_results = {}
        
        # 1. PCA
        try:
            pca = PCA(n_components=n_components)
            pca_transformed = pca.fit_transform(scaled_data)
            
            dim_reduction_results['pca'] = {
                'explained_variance': pca.explained_variance_.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'components': pca.components_.tolist(),
                'feature_importance': {
                    feature: abs(pca.components_[0][i]) 
                    for i, feature in enumerate(numeric_data.columns)
                },
                'n_components_95': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1),
                'transformed_data': pca_transformed[:, :2] if pca_transformed.shape[1] >= 2 else pca_transformed
            }
            
            # Kaiser criterion
            kaiser_threshold = 1.0
            n_kaiser = sum(pca.explained_variance_ > kaiser_threshold)
            dim_reduction_results['pca']['kaiser_components'] = int(n_kaiser)
            
        except Exception as e:
            logger.warning(f"PCA failed: {e}")
            dim_reduction_results['pca'] = {'error': str(e)}
        
        # 2. Factor Analysis (if available)
        if len(numeric_data) > 50:  # Need sufficient samples
            try:
                n_factors = min(5, len(numeric_data.columns) // 2)
                fa = FactorAnalysis(n_components=n_factors, random_state=42)
                fa_transformed = fa.fit_transform(scaled_data)
                
                dim_reduction_results['factor_analysis'] = {
                    'components': fa.components_.tolist(),
                    'noise_variance': fa.noise_variance_.tolist(),
                    'score': float(fa.score(scaled_data))
                }
            except Exception as e:
                logger.warning(f"Factor Analysis failed: {e}")
        
        # 3. t-SNE (for smaller datasets)
        if len(numeric_data) < 5000:
            try:
                tsne = TSNE(n_components=2, random_state=42)
                tsne_transformed = tsne.fit_transform(scaled_data)
                
                dim_reduction_results['tsne'] = {
                    'embedding': tsne_transformed.tolist() if len(tsne_transformed) < 1000 else 'Too large to store',
                    'kl_divergence': float(tsne.kl_divergence_)
                }
            except Exception as e:
                logger.warning(f"t-SNE failed: {e}")
        
        return dim_reduction_results
    
    def perform_clustering_analysis(self, data: pd.DataFrame,
                                  max_clusters: int = 10) -> Dict[str, Any]:
        """Perform clustering analysis to identify patterns"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Remove columns with zero variance
        numeric_data = numeric_data.loc[:, numeric_data.var() > 0]
        
        if len(numeric_data.columns) < 2:
            return {'error': 'Insufficient features for clustering'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        clustering_results = {}
        
        # 1. K-means clustering
        try:
            kmeans_scores = []
            silhouette_scores = []
            
            k_range = range(2, min(max_clusters, len(data) // 10))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scaled_data)
                
                # Calculate metrics
                inertia = kmeans.inertia_
                silhouette = silhouette_score(scaled_data, labels)
                calinski = calinski_harabasz_score(scaled_data, labels)
                
                kmeans_scores.append({
                    'k': k,
                    'inertia': float(inertia),
                    'silhouette': float(silhouette),
                    'calinski_harabasz': float(calinski)
                })
                silhouette_scores.append(silhouette)
            
            # Find optimal k using elbow method
            if len(kmeans_scores) > 0:
                optimal_k = k_range[np.argmax(silhouette_scores)]
                
                # Fit final model with optimal k
                best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                best_labels = best_kmeans.fit_predict(scaled_data)
                
                clustering_results['kmeans'] = {
                    'scores': kmeans_scores,
                    'optimal_k': int(optimal_k),
                    'cluster_labels': best_labels.tolist() if len(best_labels) < 10000 else None,
                    'cluster_centers': best_kmeans.cluster_centers_.tolist(),
                    'cluster_sizes': pd.Series(best_labels).value_counts().to_dict()
                }
        
        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}")
            clustering_results['kmeans'] = {'error': str(e)}
        
        # 2. DBSCAN (for anomaly detection)
        if len(data) < 10000:  # Limit for performance
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(scaled_data)
                
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                clustering_results['dbscan'] = {
                    'n_clusters': int(n_clusters),
                    'n_noise_points': int(n_noise),
                    'noise_ratio': float(n_noise / len(data))
                }
                
            except Exception as e:
                logger.warning(f"DBSCAN failed: {e}")
        
        return clustering_results
    
    def perform_statistical_tests(self, data: pd.DataFrame,
                                target_column: str = None,
                                task_type: str = None) -> Dict[str, Any]:
        """Perform various statistical tests"""
        
        test_results = {}
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        # 1. Test for equal variances (Levene's test)
        if target_column and task_type == 'classification':
            for feat in numeric_data.columns:
                if feat != target_column:
                    groups = [group[feat].dropna().values 
                             for name, group in data.groupby(target_column)]
                    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                        stat, p_value = stats.levene(*groups)
                        test_results[f'levene_{feat}'] = {
                            'test': "Levene's test",
                            'feature': feat,
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'equal_variance': p_value > 0.05
                        }
        
        # 2. Test for independence (if we have categorical variables)
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) >= 2:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    try:
                        contingency = pd.crosstab(data[col1], data[col2])
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
                        
                        test_results[f'chi2_{col1}_vs_{col2}'] = {
                            'test': 'Chi-square independence',
                            'variables': [col1, col2],
                            'statistic': float(chi2_stat),
                            'p_value': float(p_value),
                            'dof': int(dof),
                            'independent': p_value > 0.05
                        }
                    except Exception as e:
                        logger.warning(f"Chi-square test failed for {col1} vs {col2}: {e}")
        
        # 3. Heteroscedasticity test (if we have regression target)
        if STATSMODELS_AVAILABLE and target_column and task_type == 'regression':
            try:
                # Simple linear regression for heteroscedasticity test
                X = numeric_data.drop(columns=[target_column])
                y = numeric_data[target_column]
                
                if len(X.columns) > 0:
                    X_with_const = sm.add_constant(X.iloc[:, :5])  # Limit features
                    model = sm.OLS(y, X_with_const).fit()
                    
                    # Breusch-Pagan test
                    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X_with_const)
                    
                    test_results['breusch_pagan'] = {
                        'test': 'Breusch-Pagan',
                        'statistic': float(bp_stat),
                        'p_value': float(bp_p),
                        'homoscedastic': bp_p > 0.05
                    }
            except Exception as e:
                logger.warning(f"Heteroscedasticity test failed: {e}")
        
        return test_results
    
    def analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in the dataset"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        outlier_results = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            outliers_zscore = col_data[z_scores > 3]
            
            # Isolation Forest would go here if needed
            
            outlier_results[col] = {
                'iqr_method': {
                    'n_outliers': len(outliers_iqr),
                    'outlier_ratio': float(len(outliers_iqr) / len(col_data)),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                },
                'zscore_method': {
                    'n_outliers': len(outliers_zscore),
                    'outlier_ratio': float(len(outliers_zscore) / len(col_data))
                }
            }
        
        # Overall outlier summary
        total_iqr_outliers = sum(res['iqr_method']['n_outliers'] for res in outlier_results.values())
        total_zscore_outliers = sum(res['zscore_method']['n_outliers'] for res in outlier_results.values())
        
        outlier_results['summary'] = {
            'total_iqr_outliers': int(total_iqr_outliers),
            'total_zscore_outliers': int(total_zscore_outliers),
            'features_with_outliers': [col for col, res in outlier_results.items() 
                                      if res['iqr_method']['n_outliers'] > 0]
        }
        
        return outlier_results
    
    def check_time_series_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Time-series diagnostics for numeric columns when a DatetimeIndex is present."""
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available for time series analysis'}
        if not self._has_datetime_index(data):
            return {}

        data = data.sort_index()
        results: Dict[str, Any] = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # simple daily detector (median spacing ≈ 1 day, ±1 hour)
        def _is_daily(idx: pd.DatetimeIndex) -> bool:
            if len(idx) < 3:
                return False
            diffs = pd.Series(idx).diff().dropna()
            if diffs.empty:
                return False
            med = diffs.median()
            return abs(med - pd.Timedelta(days=1)) < pd.Timedelta(hours=1)

        is_daily = _is_daily(data.index)

        for col in list(numeric_cols)[:5]:  # limit to first 5 numeric cols
            try:
                s = data[col].dropna()
                if len(s) < 20:
                    continue

                # ADF (H0: unit root / non-stationary). p<0.05 => stationary
                adf_stat, adf_p, adf_lags, _, adf_crit = adfuller(s, autolag='AIC')

                # KPSS (H0: stationary). p<0.05 => non-stationary
                try:
                    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression='c', nlags='auto')
                except Exception:
                    kpss_stat = kpss_p = kpss_lags = None
                    kpss_crit = {}

                col_res: Dict[str, Any] = {
                    'adf_statistic': float(adf_stat),
                    'adf_pvalue': float(adf_p),
                    'is_stationary_adf': bool(adf_p < 0.05),
                    'kpss_statistic': float(kpss_stat) if kpss_stat is not None else None,
                    'kpss_pvalue': float(kpss_p) if kpss_p is not None else None,
                    'is_stationary_kpss': bool(kpss_p is not None and kpss_p > 0.05),
                    'lags_used': int(adf_lags),
                    'critical_values': {k: float(v) for k, v in adf_crit.items()},
                }

                # Seasonal decomposition: if daily, try 7 first, else 30 (need ~>= 2*period samples)
                seasonal_period = None
                if is_daily and len(s) >= 7 * 2 + 1:
                    seasonal_period = 7
                elif is_daily and len(s) >= 30 * 2 + 1:
                    seasonal_period = 30

                if seasonal_period:
                    dec = seasonal_decompose(s, model='additive', period=seasonal_period)
                    col_res['seasonality'] = {
                        'seasonal_strength': float(np.std(dec.seasonal) / (np.std(s) + 1e-9)),
                        'trend_strength': float(np.std(pd.Series(dec.trend).dropna()) / (np.std(s) + 1e-9)),
                    }
                    col_res['seasonal_period'] = int(seasonal_period)

                # ACF for daily lags 1..10 (or up to N//4)
                max_lag = int(min(10, max(1, len(s) // 4)))
                acf_vals = acf(s, nlags=max_lag, fft=True, missing='conservative')
                col_res['acf_by_lag'] = {int(l): float(acf_vals[l]) for l in range(1, max_lag + 1)}
                top = sorted(col_res['acf_by_lag'].items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                col_res['acf_top_lags'] = [{'lag': int(k), 'acf': float(v)} for k, v in top]
                col_res['lag_unit'] = 'days' if is_daily else 'index_steps'

                results[col] = col_res

            except Exception as e:
                logger.warning(f"Time series analysis failed for {col}: {e}")

        return results

    
    def _identify_distribution(self, data: np.ndarray) -> str:
        """Identify the likely distribution type"""
        
        # Simple heuristic-based identification
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            return "normal"
        elif skewness > 2:
            return "right_skewed"
        elif skewness < -2:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif len(np.unique(data)) < 20:
            return "discrete"
        else:
            return "unknown"
    
    def _has_datetime_index(self, data: pd.DataFrame) -> bool:
        """Check if dataframe has datetime index"""
        return isinstance(data.index, pd.DatetimeIndex)
    
    def generate_statistical_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights from statistical analysis"""
        
        insights = []
        
        # Basic statistics insights
        if 'basic_statistics' in results:
            basic = results['basic_statistics']
            if basic['numeric']['zero_variance']:
                zero_var_features = [k for k, v in basic['numeric']['zero_variance'].items() if v]
                if zero_var_features:
                    insights.append(f"[!] Found {len(zero_var_features)} constant features that should be removed")
        
        # Distribution insights
        if 'distribution_analysis' in results:
            non_normal = [k for k, v in results['distribution_analysis'].items() 
                         if not v['normality']['is_normal']]
            if non_normal:
                insights.append(f"[▊] {len(non_normal)} features have non-normal distributions - consider transformations")
        
        # Correlation insights
        if 'correlation_analysis' in results:
            high_corr = results['correlation_analysis'].get('high_correlations', [])
            if high_corr:
                insights.append(f"🔗 Found {len(high_corr)} highly correlated feature pairs (>0.8) - consider removing redundant features")
        
        # Multicollinearity insights
        if 'multicollinearity' in results:
            high_vif = results['multicollinearity'].get('high_vif_features', [])
            if high_vif:
                insights.append(f"[▲] {len(high_vif)} features show high multicollinearity (VIF>10)")
        
        # PCA insights
        if 'dimensionality_reduction' in results and 'pca' in results['dimensionality_reduction']:
            pca = results['dimensionality_reduction']['pca']
            if 'n_components_95' in pca:
                insights.append(f"[◎] PCA: {pca['n_components_95']} components explain 95% of variance")
        
        # Clustering insights
        if 'clustering_analysis' in results and 'kmeans' in results['clustering_analysis']:
            kmeans = results['clustering_analysis']['kmeans']
            if 'optimal_k' in kmeans:
                insights.append(f"[◯] Optimal number of clusters: {kmeans['optimal_k']} (by silhouette score)")
        
        # Outlier insights
        if 'outlier_analysis' in results:
            outlier_features = results['outlier_analysis']['summary']['features_with_outliers']
            if outlier_features:
                insights.append(f"⚡ {len(outlier_features)} features contain outliers - consider robust scaling")
        
        return insights
    
    def create_statistical_report_data(self) -> Dict[str, Any]:
        """Create data for statistical report visualization"""
        
        if not self.results:
            return {}
        
        report_data = {
            'summary': self.results.get('basic_statistics', {}).get('dataset', {}),
            'insights': self.results.get('insights', []),
            'key_findings': self._extract_key_findings(),
            'visualizations': self._prepare_visualization_data()
        }
        
        return report_data
    
    def _extract_key_findings(self) -> List[Dict[str, Any]]:
        """Extract key findings from analysis results"""
        
        findings = []
        
        # Top correlated features with target
        if 'correlation_analysis' in self.results:
            target_corr = self.results['correlation_analysis'].get('target_correlations', {})
            if target_corr:
                top_features = list(target_corr.items())[:5]
                findings.append({
                    'type': 'target_correlation',
                    'title': 'Top Features Correlated with Target',
                    'data': [(feat, data['pearson']) for feat, data in top_features]
                })
        
        # PCA variance explained
        if 'dimensionality_reduction' in self.results:
            pca = self.results['dimensionality_reduction'].get('pca', {})
            if 'explained_variance_ratio' in pca:
                findings.append({
                    'type': 'pca_variance',
                    'title': 'PCA Variance Explained',
                    'data': pca['explained_variance_ratio'][:10]
                })
        
        return findings
    
    def _prepare_visualization_data(self) -> Dict[str, Any]:
        """Prepare data for statistical visualizations"""
        
        viz_data = {}
        
        # Prepare correlation matrix data
        if 'correlation_analysis' in self.results:
            viz_data['correlation_matrix'] = self.results['correlation_analysis'].get('pearson', {})
        
        # Prepare PCA data
        if 'dimensionality_reduction' in self.results:
            pca = self.results['dimensionality_reduction'].get('pca', {})
            if 'transformed_data' in pca:
                viz_data['pca_scatter'] = pca['transformed_data']
        
        # Prepare distribution data
        if 'distribution_analysis' in self.results:
            viz_data['distributions'] = self.results['distribution_analysis']
        
        return viz_data
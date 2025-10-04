"""
Data Quality Checker Module for AquaVista v6.0
=============================================
Performs comprehensive data quality assessments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
from datetime import datetime

# Import custom modules
from modules.config import Config
from modules.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class DataQualityChecker:
    """Performs comprehensive data quality checks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.quality_report = {}
        
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing quality metrics and recommendations
        """
        logger.info(f"Starting data quality check for dataframe with shape {df.shape}")
        
        self.quality_report = {
            'basic_info': self._get_basic_info(df),
            'missing_analysis': self._analyze_missing_values(df),
            'duplicate_analysis': self._analyze_duplicates(df),
            'data_types': self._analyze_data_types(df),
            'statistical_summary': self._get_statistical_summary(df),
            'outlier_analysis': self._detect_outliers(df),
            'correlation_analysis': self._analyze_correlations(df),
            'data_quality_issues': [],
            'recommendations': [],
            'overall_quality_score': 0
        }
        
        # Calculate overall quality score
        self._calculate_quality_score()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Identify recommended features
        self._identify_recommended_features(df)
        
        logger.info(f"Data quality check complete. Overall score: {self.quality_report['overall_quality_score']:.1f}%")
        
        return self.quality_report
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataframe"""
        return {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'column_count': len(df.columns),
            'row_count': len(df),
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the dataframe"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        analysis = {
            'total_missing': missing_counts.sum(),
            'missing_percentage': (missing_counts.sum() / (df.shape[0] * df.shape[1])) * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'columns_with_missing_pct': missing_percentages[missing_percentages > 0].to_dict(),
            'columns_all_missing': df.columns[df.isnull().all()].tolist(),
            'rows_with_missing': df.isnull().any(axis=1).sum(),
            'complete_rows': (~df.isnull().any(axis=1)).sum()
        }
        
        # Identify patterns in missing data
        if analysis['total_missing'] > 0:
            analysis['missing_patterns'] = self._identify_missing_patterns(df)
        
        return analysis
    
    def _identify_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns in missing data"""
        patterns = {}
        
        # Check for columns that are missing together
        missing_mask = df.isnull()
        correlations = missing_mask.corr()
        
        # Find highly correlated missing patterns
        high_corr_pairs = []
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                if abs(correlations.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'columns': [correlations.columns[i], correlations.columns[j]],
                        'correlation': correlations.iloc[i, j]
                    })
        
        patterns['correlated_missing'] = high_corr_pairs
        
        return patterns
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows in the dataframe"""
        duplicate_mask = df.duplicated()
        
        analysis = {
            'duplicate_rows': duplicate_mask.sum(),
            'duplicate_percentage': (duplicate_mask.sum() / len(df)) * 100,
            'first_duplicate_index': df[duplicate_mask].index.tolist()[:5] if duplicate_mask.any() else []
        }
        
        # Check for duplicate columns
        duplicate_cols = []
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if df[col1].equals(df[col2]):
                    duplicate_cols.append((col1, col2))
        
        analysis['duplicate_columns'] = duplicate_cols
        
        return analysis
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and identify potential issues"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        analysis = {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'mixed_type_columns': [],
            'high_cardinality_columns': [],
            'low_cardinality_numeric': [],
            'potential_datetime_columns': []
        }
        
        # Check for mixed types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='coerce').notna().sum()
                    if pd.to_numeric(df[col], errors='coerce').notna().sum() > len(df) * 0.5:
                        analysis['mixed_type_columns'].append(col)
                except:
                    pass
        
        # Check cardinality
        for col in categorical_cols:
            cardinality = df[col].nunique()
            if cardinality > len(df) * 0.5:
                analysis['high_cardinality_columns'].append({
                    'column': col,
                    'unique_values': cardinality,
                    'percentage': (cardinality / len(df)) * 100
                })
        
        # Check for categorical-like numeric columns
        for col in numeric_cols:
            if df[col].nunique() < 20:
                analysis['low_cardinality_numeric'].append({
                    'column': col,
                    'unique_values': df[col].nunique(),
                    'values': sorted(df[col].unique())[:10]
                })
        
        return analysis
    
    def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary of numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {}
        
        summary = {
            'numeric_summary': numeric_df.describe().to_dict(),
            'skewness': numeric_df.skew().to_dict(),
            'kurtosis': numeric_df.kurtosis().to_dict()
        }
        
        # Identify highly skewed columns
        summary['highly_skewed'] = [
            col for col, skew in summary['skewness'].items() 
            if abs(skew) > 1
        ]
        
        return summary
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = {}
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min_outlier': numeric_df[col][outlier_mask].min(),
                    'max_outlier': numeric_df[col][outlier_mask].max()
                }
        
        return outliers
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric features"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {}
        
        corr_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'high_correlations': high_correlations,
            'correlation_matrix': corr_matrix.to_dict()
        }
    
    def _calculate_quality_score(self):
        """Calculate overall data quality score"""
        score = 100.0
        
        # Deduct points for various issues
        missing_pct = self.quality_report['missing_analysis']['missing_percentage']
        score -= min(missing_pct * 2, 30)  # Max 30 point deduction
        
        duplicate_pct = self.quality_report['duplicate_analysis']['duplicate_percentage']
        score -= min(duplicate_pct * 1.5, 20)  # Max 20 point deduction
        
        # Deduct for high cardinality
        high_card_count = len(self.quality_report['data_types']['high_cardinality_columns'])
        score -= min(high_card_count * 5, 15)  # Max 15 point deduction
        
        # Deduct for outliers
        outlier_cols = len(self.quality_report['outlier_analysis'])
        score -= min(outlier_cols * 2, 10)  # Max 10 point deduction
        
        # Deduct for mixed types
        mixed_types = len(self.quality_report['data_types']['mixed_type_columns'])
        score -= min(mixed_types * 3, 10)  # Max 10 point deduction
        
        # Ensure score is between 0 and 100
        self.quality_report['overall_quality_score'] = max(0, min(100, score))
        
        # Store deductions for transparency
        self.quality_report['score_deductions'] = {
            'missing_data': min(missing_pct * 2, 30),
            'duplicates': min(duplicate_pct * 1.5, 20),
            'high_cardinality': min(high_card_count * 5, 15),
            'outliers': min(outlier_cols * 2, 10),
            'mixed_types': min(mixed_types * 3, 10)
        }
    
    def _generate_recommendations(self):
        """Generate recommendations based on quality analysis"""
        recommendations = []
        
        # Missing data recommendations
        if self.quality_report['missing_analysis']['missing_percentage'] > 5:
            recommendations.append(
                f"Consider handling missing values ({self.quality_report['missing_analysis']['missing_percentage']:.1f}% missing)"
            )
            
        # Duplicate recommendations
        if self.quality_report['duplicate_analysis']['duplicate_rows'] > 0:
            recommendations.append(
                f"Remove {self.quality_report['duplicate_analysis']['duplicate_rows']} duplicate rows"
            )
            
        # High cardinality recommendations
        for col_info in self.quality_report['data_types']['high_cardinality_columns']:
            recommendations.append(
                f"Column '{col_info['column']}' has high cardinality ({col_info['unique_values']} unique values)"
            )
            
        # Outlier recommendations
        if len(self.quality_report['outlier_analysis']) > 0:
            recommendations.append(
                f"Investigate outliers in {len(self.quality_report['outlier_analysis'])} columns"
            )
            
        # Correlation recommendations
        if 'high_correlations' in self.quality_report['correlation_analysis']:
            for corr in self.quality_report['correlation_analysis']['high_correlations']:
                recommendations.append(
                    f"High correlation ({corr['correlation']:.2f}) between '{corr['feature1']}' and '{corr['feature2']}'"
                )
        
        self.quality_report['recommendations'] = recommendations
    
    def _identify_recommended_features(self, df: pd.DataFrame):
        """Identify features recommended for modeling"""
        recommended = []
        not_recommended = []
        
        for col in df.columns:
            # Skip if too many missing values
            if col in self.quality_report['missing_analysis']['columns_with_missing_pct']:
                if self.quality_report['missing_analysis']['columns_with_missing_pct'][col] > 50:
                    not_recommended.append(col)
                    continue
            
            # Skip if constant
            if df[col].nunique() == 1:
                not_recommended.append(col)
                continue
                
            # Skip if too high cardinality for categorical
            if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.9:
                not_recommended.append(col)
                continue
            
            recommended.append(col)
        
        self.quality_report['recommended_features'] = recommended
        self.quality_report['not_recommended_features'] = not_recommended
        
        # Add summary metrics for quick reference
        self.quality_report['missing_percentage'] = self.quality_report['missing_analysis']['missing_percentage']
        self.quality_report['duplicate_rows'] = self.quality_report['duplicate_analysis']['duplicate_rows']
        self.quality_report['high_cardinality_features'] = [
            col['column'] for col in self.quality_report['data_types']['high_cardinality_columns']
        ]
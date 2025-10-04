"""
Model Ranking Module for AquaVista v7.0
======================================
Comprehensive model ranking system that evaluates models across multiple criteria:
- Performance metrics
- Interpretability and reliability
- Training efficiency
- Robustness and stability
- Data-specific suitability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import warnings

class ModelRankingSystem:
    """
    Comprehensive model ranking system that evaluates models across multiple criteria
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the ranking system with customizable weights
        
        Args:
            weights: Dictionary of criteria weights. Default provides balanced scoring.
        """
        self.default_weights = {
            'performance': 0.35,           # Primary metric performance
            'stability': 0.20,             # Cross-validation stability
            'interpretability': 0.20,      # SHAP availability and feature importance
            'efficiency': 0.15,            # Training time and resource usage
            'robustness': 0.10            # Overfitting and generalization
        }
        
        self.weights = weights or self.default_weights
        self.ranking_results = {}
        
    def rank_models(self, training_results: Dict, 
                   dataset_info: Dict = None,
                   use_case_priority: str = 'balanced') -> Dict[str, Any]:
        """
        Rank models based on comprehensive multi-criteria evaluation
        
        Args:
            training_results: Results from ModelManager.train_models()
            dataset_info: Information about dataset characteristics
            use_case_priority: 'performance', 'interpretability', 'efficiency', or 'balanced'
            
        Returns:
            Dictionary with rankings, scores, and explanations
        """
        
        # Adjust weights based on use case priority
        weights = self._adjust_weights_for_use_case(use_case_priority)
        
        models = training_results.get('models', {})
        if not models:
            return {'error': 'No models to rank'}
        
        # Calculate scores for each criterion
        scores = {}
        explanations = {}
        
        for model_name, model_data in models.items():
            try:
                model_scores = self._calculate_model_scores(
                    model_name, model_data, training_results, dataset_info
                )
                scores[model_name] = model_scores
                explanations[model_name] = self._generate_model_explanation(
                    model_name, model_scores, model_data
                )
            except Exception as e:
                # Handle failed models gracefully
                scores[model_name] = self._get_default_scores()
                explanations[model_name] = {
                    'summary': f"Failed to evaluate: {str(e)}",
                    'strengths': [],
                    'weaknesses': ['Model evaluation failed'],
                    'recommendations': ['Review model training logs']
                }
        
        # Calculate weighted final scores
        final_scores = {}
        for model_name, model_scores in scores.items():
            final_score = sum(
                model_scores[criterion] * weights[criterion] 
                for criterion in weights.keys()
            )
            final_scores[model_name] = final_score
        
        # Create rankings
        ranked_models = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate comprehensive results
        results = {
            'rankings': self._create_ranking_table(ranked_models, scores, explanations),
            'methodology': self._explain_methodology(weights),
            'recommendations': self._generate_overall_recommendations(
                ranked_models, scores, explanations, training_results
            ),
            'weights_used': weights,
            'use_case_priority': use_case_priority
        }
        
        self.ranking_results = results
        return results
    
    def _adjust_weights_for_use_case(self, use_case: str) -> Dict[str, float]:
        """Adjust weights based on use case priority"""
        if use_case == 'performance':
            return {
                'performance': 0.50,
                'stability': 0.20,
                'interpretability': 0.10,
                'efficiency': 0.10,
                'robustness': 0.10
            }
        elif use_case == 'interpretability':
            return {
                'performance': 0.25,
                'stability': 0.15,
                'interpretability': 0.40,
                'efficiency': 0.10,
                'robustness': 0.10
            }
        elif use_case == 'efficiency':
            return {
                'performance': 0.30,
                'stability': 0.15,
                'interpretability': 0.15,
                'efficiency': 0.30,
                'robustness': 0.10
            }
        else:  # balanced
            return self.default_weights.copy()
    
    def _calculate_model_scores(self, model_name: str, model_data: Dict, 
                              training_results: Dict, dataset_info: Dict) -> Dict[str, float]:
        """Calculate normalized scores (0-1) for each criterion"""
        
        scores = {}
        
        # 1. Performance Score
        scores['performance'] = self._calculate_performance_score(
            model_data, training_results
        )
        
        # 2. Stability Score
        scores['stability'] = self._calculate_stability_score(model_data)
        
        # 3. Interpretability Score
        scores['interpretability'] = self._calculate_interpretability_score(
            model_name, model_data
        )
        
        # 4. Efficiency Score
        scores['efficiency'] = self._calculate_efficiency_score(
            model_data, training_results
        )
        
        # 5. Robustness Score
        scores['robustness'] = self._calculate_robustness_score(model_data)
        
        return scores
    
    def _calculate_performance_score(self, model_data: Dict, training_results: Dict) -> float:
        """Calculate performance score based on test metrics and baseline comparison"""
        try:
            primary_metric = training_results.get('primary_metric', 'r2_score')
            test_score = model_data.get('test_scores', {}).get(primary_metric, 0)
            baseline = training_results.get('baseline_score', 0)
            
            if baseline == 0:
                return min(test_score, 1.0) if test_score >= 0 else 0.0
            
            # Score based on improvement over baseline
            if primary_metric in ['r2_score', 'accuracy']:
                # Higher is better
                improvement = (test_score - baseline) / max(abs(baseline), 0.1)
                return min(max(improvement, 0) / 2 + 0.5, 1.0)  # Scale to 0-1
            else:
                # Lower is better (error metrics)
                if test_score <= baseline:
                    improvement = (baseline - test_score) / max(baseline, 0.1)
                    return min(improvement, 1.0)
                else:
                    return max(0.5 - (test_score - baseline) / max(baseline, 0.1), 0.0)
                    
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, model_data: Dict) -> float:
        """Calculate stability score based on CV performance"""
        try:
            cv_scores = model_data.get('cv_scores')
            if not cv_scores:
                return 0.5  # Neutral score if no CV data
            
            cv_mean = cv_scores.get('mean', 0)
            cv_std = cv_scores.get('std', 0)
            
            if cv_std == 0:
                return 1.0  # Perfect stability
            
            # Lower standard deviation is better
            # Coefficient of variation as stability metric
            cv_coefficient = cv_std / max(abs(cv_mean), 0.01)
            stability_score = max(0, 1 - cv_coefficient * 2)  # Scale appropriately
            
            return min(stability_score, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_interpretability_score(self, model_name: str, model_data: Dict) -> float:
        """Calculate interpretability score"""
        score = 0.0
        
        # Base interpretability by model type
        interpretable_models = {
            'Linear Regression': 0.9,
            'Logistic Regression': 0.9,
            'Ridge': 0.9,
            'Lasso': 0.9,
            'Decision Tree': 0.8,
            'Random Forest': 0.7,
            'Extra Trees': 0.7,
            'Gradient Boosting': 0.6
        }
        
        base_score = interpretable_models.get(model_name, 0.4)
        score += base_score * 0.4
        
        # SHAP availability bonus
        if model_data.get('shap_available', False):
            score += 0.3
            
            # SHAP validation bonus
            if model_data.get('shap_validated', False):
                score += 0.2
            elif model_data.get('shap_math_valid', False):
                score += 0.1
        
        # Feature importance availability
        feature_importance = model_data.get('feature_importance', {})
        if feature_importance and feature_importance.get('importances'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_efficiency_score(self, model_data: Dict, training_results: Dict) -> float:
        """Calculate efficiency score based on training time"""
        try:
            training_time = model_data.get('training_time', 0)
            
            if training_time <= 0:
                return 0.5  # Neutral if no timing data
            
            # Get all training times for comparison
            all_times = [
                data.get('training_time', 0) 
                for data in training_results.get('models', {}).values()
                if data.get('training_time', 0) > 0
            ]
            
            if not all_times:
                return 0.5
            
            min_time = min(all_times)
            max_time = max(all_times)
            
            if max_time == min_time:
                return 1.0
            
            # Inverse relationship: faster = better score
            normalized_time = (training_time - min_time) / (max_time - min_time)
            efficiency_score = 1.0 - normalized_time
            
            return max(efficiency_score, 0.1)  # Minimum score for any working model
            
        except Exception:
            return 0.5
    
    def _calculate_robustness_score(self, model_data: Dict) -> float:
        """Calculate robustness score based on overfitting and generalization"""
        try:
            train_scores = model_data.get('train_scores', {})
            test_scores = model_data.get('test_scores', {})
            
            if not train_scores or not test_scores:
                return 0.5
            
            # Use primary metric (usually r2_score for regression, accuracy for classification)
            train_metric = None
            test_metric = None
            
            for metric in ['r2_score', 'accuracy']:
                if metric in train_scores and metric in test_scores:
                    train_metric = train_scores[metric]
                    test_metric = test_scores[metric]
                    break
            
            if train_metric is None or test_metric is None:
                return 0.5
            
            # Calculate overfitting penalty
            if train_metric == 0:
                return 0.5
            
            # For metrics where higher is better
            if train_metric >= test_metric:
                overfitting_ratio = (train_metric - test_metric) / max(abs(train_metric), 0.01)
                robustness_score = max(0, 1 - overfitting_ratio * 2)  # Penalty for overfitting
            else:
                # Test score higher than train score (unusual but possible)
                robustness_score = 0.9  # Slightly lower for this unusual case
            
            return min(robustness_score, 1.0)
            
        except Exception:
            return 0.5
    
    def _generate_model_explanation(self, model_name: str, scores: Dict, 
                                  model_data: Dict) -> Dict[str, Any]:
        """Generate detailed explanation for model ranking"""
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Performance analysis
        if scores['performance'] > 0.8:
            strengths.append("Excellent predictive performance")
        elif scores['performance'] > 0.6:
            strengths.append("Good predictive performance")
        else:
            weaknesses.append("Below-average predictive performance")
            recommendations.append("Consider feature engineering or different algorithms")
        
        # Stability analysis
        if scores['stability'] > 0.8:
            strengths.append("High cross-validation stability")
        elif scores['stability'] < 0.5:
            weaknesses.append("Poor cross-validation stability")
            recommendations.append("Model may be sensitive to data variations")
        
        # Interpretability analysis
        if scores['interpretability'] > 0.7:
            strengths.append("Good interpretability with reliable feature insights")
        elif scores['interpretability'] < 0.4:
            weaknesses.append("Limited interpretability")
            if not model_data.get('shap_available', False):
                recommendations.append("SHAP analysis failed - consider simpler models for interpretation")
        
        # Efficiency analysis
        if scores['efficiency'] > 0.8:
            strengths.append("Fast training time")
        elif scores['efficiency'] < 0.3:
            weaknesses.append("Slow training time")
            recommendations.append("Consider for batch processing only")
        
        # Robustness analysis
        if scores['robustness'] < 0.5:
            train_scores = model_data.get('train_scores', {})
            test_scores = model_data.get('test_scores', {})
            if train_scores and test_scores:
                train_r2 = train_scores.get('r2_score', 0)
                test_r2 = test_scores.get('r2_score', 0)
                if train_r2 - test_r2 > 0.2:
                    weaknesses.append("Shows signs of overfitting")
                    recommendations.append("Consider regularization or ensemble methods")
        
        # Model-specific insights
        model_insights = self._get_model_specific_insights(model_name, model_data)
        strengths.extend(model_insights['strengths'])
        weaknesses.extend(model_insights['weaknesses'])
        recommendations.extend(model_insights['recommendations'])
        
        # Generate summary
        avg_score = np.mean(list(scores.values()))
        if avg_score > 0.8:
            summary = f"Excellent overall model with strong performance across multiple criteria"
        elif avg_score > 0.6:
            summary = f"Good model with solid performance in most areas"
        elif avg_score > 0.4:
            summary = f"Average model with mixed performance across criteria"
        else:
            summary = f"Below-average model with several areas needing improvement"
        
        return {
            'summary': summary,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'detailed_scores': scores
        }
    
    def _get_model_specific_insights(self, model_name: str, model_data: Dict) -> Dict[str, List[str]]:
        """Get model-specific insights based on model type and performance"""
        insights = {'strengths': [], 'weaknesses': [], 'recommendations': []}
        
        if 'Decision Tree' in model_name:
            train_r2 = model_data.get('train_scores', {}).get('r2_score', 0)
            test_r2 = model_data.get('test_scores', {}).get('r2_score', 0)
            
            if train_r2 > 0.95 and test_r2 < 0.7:
                insights['weaknesses'].append("Severe overfitting detected")
                insights['recommendations'].append("Use ensemble methods or pruning")
            
            insights['strengths'].append("Highly interpretable with clear decision paths")
            
        elif model_name in ['Random Forest', 'Extra Trees']:
            insights['strengths'].append("Handles feature interactions well")
            insights['strengths'].append("Robust to outliers")
            
        elif 'Bagging' in model_name:
            if model_data.get('hyperparameter_optimized', False):
                insights['weaknesses'].append("Feature interpretation may be unstable due to hyperparameter sensitivity")
                insights['recommendations'].append("Validate feature importance across different parameter settings")
            
        elif model_name in ['Ridge', 'Lasso', 'Linear Regression']:
            insights['strengths'].append("Linear relationships are easily interpretable")
            if 'Lasso' in model_name:
                insights['strengths'].append("Automatic feature selection capability")
        
        return insights
    
    def _create_ranking_table(self, ranked_models: List[Tuple], scores: Dict, 
                            explanations: Dict) -> List[Dict]:
        """Create detailed ranking table"""
        ranking_table = []
        
        for rank, (model_name, final_score) in enumerate(ranked_models, 1):
            model_scores = scores[model_name]
            explanation = explanations[model_name]
            
            ranking_table.append({
                'rank': rank,
                'model': model_name,
                'overall_score': round(final_score, 3),
                'performance_score': round(model_scores['performance'], 3),
                'stability_score': round(model_scores['stability'], 3),
                'interpretability_score': round(model_scores['interpretability'], 3),
                'efficiency_score': round(model_scores['efficiency'], 3),
                'robustness_score': round(model_scores['robustness'], 3),
                'summary': explanation['summary'],
                'key_strengths': explanation['strengths'][:2],  # Top 2 strengths
                'main_weakness': explanation['weaknesses'][0] if explanation['weaknesses'] else None,
                'top_recommendation': explanation['recommendations'][0] if explanation['recommendations'] else None
            })
        
        return ranking_table
    
    def _explain_methodology(self, weights: Dict) -> Dict[str, Any]:
        """Explain the ranking methodology"""
        return {
            'description': 'Models are ranked using a weighted multi-criteria scoring system',
            'criteria_weights': weights,
            'scoring_details': {
                'performance': 'Based on test score improvement over baseline',
                'stability': 'Cross-validation consistency (lower std dev = higher score)',
                'interpretability': 'SHAP availability, feature importance, model transparency',
                'efficiency': 'Training time relative to other models',
                'robustness': 'Generalization ability (train vs test score gap)'
            },
            'score_range': 'All scores normalized to 0-1 scale, with 1 being optimal'
        }
    
    def _generate_overall_recommendations(self, ranked_models: List[Tuple], 
                                        scores: Dict, explanations: Dict,
                                        training_results: Dict) -> List[str]:
        """Generate overall recommendations based on ranking analysis"""
        recommendations = []
        
        if not ranked_models:
            return ["No models available for recommendation"]
        
        top_model = ranked_models[0][0]
        top_score = ranked_models[0][1]
        
        # Top model recommendation
        if top_score > 0.8:
            recommendations.append(f"Strong recommendation: {top_model} shows excellent overall performance")
        elif top_score > 0.6:
            recommendations.append(f"Good choice: {top_model} provides solid performance across criteria")
        else:
            recommendations.append(f"Cautious recommendation: {top_model} is best available but has limitations")
        
        # Use case specific recommendations
        performance_leader = max(scores.items(), key=lambda x: x[1]['performance'])
        interpretability_leader = max(scores.items(), key=lambda x: x[1]['interpretability'])
        efficiency_leader = max(scores.items(), key=lambda x: x[1]['efficiency'])
        
        if performance_leader[0] != top_model:
            recommendations.append(f"For maximum accuracy: Consider {performance_leader[0]}")
        
        if interpretability_leader[0] != top_model:
            recommendations.append(f"For feature interpretation: Consider {interpretability_leader[0]}")
        
        if efficiency_leader[0] != top_model:
            recommendations.append(f"For fast deployment: Consider {efficiency_leader[0]}")
        
        # Data quality insights
        baseline = training_results.get('baseline_score', 0)
        best_performance = max(scores[model]['performance'] for model in scores)
        
        if best_performance < 0.3:
            recommendations.append("⚠️ All models show poor performance - consider data quality review")
        
        # Model diversity insights
        if len(ranked_models) > 1:
            score_gap = ranked_models[0][1] - ranked_models[1][1]
            if score_gap < 0.05:
                recommendations.append("Multiple models show similar performance - consider ensemble approaches")
        
        return recommendations
    
    def _get_default_scores(self) -> Dict[str, float]:
        """Return default scores for failed models"""
        return {
            'performance': 0.0,
            'stability': 0.0,
            'interpretability': 0.0,
            'efficiency': 0.0,
            'robustness': 0.0
        }
    
    def export_ranking_report(self, filepath: str = None) -> str:
        """Export detailed ranking report"""
        if not self.ranking_results:
            return "No ranking results available. Run rank_models() first."
        
        report_lines = []
        results = self.ranking_results
        
        # Header
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE MODEL RANKING REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Use Case Priority: {results['use_case_priority']}")
        report_lines.append("")
        
        # Methodology
        report_lines.append("RANKING METHODOLOGY:")
        methodology = results['methodology']
        for criterion, weight in methodology['criteria_weights'].items():
            report_lines.append(f"  {criterion.title()}: {weight:.1%} weight - {methodology['scoring_details'][criterion]}")
        report_lines.append("")
        
        # Rankings table
        report_lines.append("MODEL RANKINGS:")
        report_lines.append("-" * 80)
        
        for model_info in results['rankings']:
            report_lines.append(f"#{model_info['rank']} - {model_info['model']}")
            report_lines.append(f"   Overall Score: {model_info['overall_score']:.3f}")
            report_lines.append(f"   Performance: {model_info['performance_score']:.3f} | "
                               f"Stability: {model_info['stability_score']:.3f} | "
                               f"Interpretability: {model_info['interpretability_score']:.3f}")
            report_lines.append(f"   Efficiency: {model_info['efficiency_score']:.3f} | "
                               f"Robustness: {model_info['robustness_score']:.3f}")
            report_lines.append(f"   Summary: {model_info['summary']}")
            
            if model_info['key_strengths']:
                report_lines.append(f"   Strengths: {', '.join(model_info['key_strengths'])}")
            
            if model_info['main_weakness']:
                report_lines.append(f"   Main Weakness: {model_info['main_weakness']}")
                
            report_lines.append("")
        
        # Overall recommendations
        report_lines.append("OVERALL RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            report_lines.append(f"{i}. {rec}")
        
        report_text = "\n".join(report_lines)
        
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(report_text)
                return f"Report exported to {filepath}"
            except Exception as e:
                return f"Failed to export report: {str(e)}"
        
        return report_text


# Usage example and integration with existing code
def integrate_ranking_with_training_results(training_results: Dict, 
                                          use_case: str = 'balanced') -> Dict:
    """
    Integrate comprehensive ranking with existing training results
    
    Args:
        training_results: Output from ModelManager.train_models()
        use_case: 'performance', 'interpretability', 'efficiency', or 'balanced'
    
    Returns:
        Enhanced results with comprehensive rankings
    """
    
    ranking_system = ModelRankingSystem()
    ranking_results = ranking_system.rank_models(training_results, use_case_priority=use_case)
    
    # Add ranking results to training results
    enhanced_results = training_results.copy()
    enhanced_results['comprehensive_ranking'] = ranking_results
    
    return enhanced_results
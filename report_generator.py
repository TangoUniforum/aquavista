"""
Report Generator Module for AquaVista v6.0
=========================================
Generates comprehensive reports from analysis results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import logging

from modules.config import Config
from modules.logging_config import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates reports for AquaVista platform"""
    
    def __init__(self, config: Config):
        self.config = config
        self.report_dir = config.export.output_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, 
                       data_info: Dict[str, Any],
                       quality_report: Optional[Dict[str, Any]] = None,
                       processing_report: Optional[Dict[str, Any]] = None,
                       training_results: Optional[Dict[str, Any]] = None,
                       statistical_results: Optional[Dict[str, Any]] = None,
                       best_model_name: Optional[str] = None) -> Path:
        """Generate comprehensive HTML report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"report_{timestamp}.html"
        
        # Generate HTML content
        html_content = self._generate_html_report(
            data_info, quality_report, processing_report,
            training_results, statistical_results, best_model_name
        )
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {report_path}")
        return report_path
    
    def _generate_html_report(self, data_info, quality_report, processing_report,
                            training_results, statistical_results, best_model_name):
        """Generate HTML report content"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AquaVista Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .info-box {{
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 1rem;
            margin: 1rem 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌊 AquaVista Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>[▊] Dataset Information</h2>
        <div class="info-box">
            <p><strong>Shape:</strong> {data_info.get('shape', 'N/A')}</p>
            <p><strong>Columns:</strong> {len(data_info.get('columns', []))}</p>
            <p><strong>Target Variable:</strong> {data_info.get('target', 'Not specified')}</p>
        </div>
"""

        # Data Quality Section
        if quality_report:
            html += f"""
        <h2>[◯] Data Quality Assessment</h2>
        <div class="metric">
            <strong>Quality Score:</strong> {quality_report.get('overall_quality_score', 0):.1f}%
        </div>
        <div class="metric">
            <strong>Missing Data:</strong> {quality_report.get('missing_percentage', 0):.1f}%
        </div>
        <div class="metric">
            <strong>Duplicate Rows:</strong> {quality_report.get('duplicate_rows', 0)}
        </div>
"""

        # Model Training Results
        if training_results and training_results.get('models'):
            html += f"""
        <h2>🤖 Model Training Results</h2>
        <div class="info-box">
            <p><strong>Best Model:</strong> {best_model_name or training_results.get('best_model', 'N/A')}</p>
            <p><strong>Best Score:</strong> {training_results.get('best_score', 0):.4f}</p>
            <p><strong>Total Training Time:</strong> {training_results.get('total_time', 0):.2f} seconds</p>
        </div>
        
        <h3>Model Performance Summary</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>Training Time (s)</th>
                <th>Test Score</th>
            </tr>
"""
            for model_name, model_data in training_results['models'].items():
                primary_metric = training_results.get('primary_metric', 'score')
                score = model_data['test_scores'].get(primary_metric, 0)
                html += f"""
            <tr>
                <td>{model_name}</td>
                <td>{model_data.get('training_time', 0):.2f}</td>
                <td>{score:.4f}</td>
            </tr>
"""
            html += "</table>"

        # Statistical Analysis
        if statistical_results and statistical_results.get('insights'):
            html += """
        <h2>[▲] Statistical Insights</h2>
        <ul>
"""
            for insight in statistical_results['insights']:
                html += f"            <li>{insight}</li>\n"
            html += "        </ul>"

        # Close HTML
        html += """
    </div>
</body>
</html>
"""
        return html
    
    # ADD THESE METHODS TO YOUR ReportGenerator CLASS IN report_generator.py
    # Add them around line 150-180, before the end of the class

    def _add_comprehensive_ranking_section(self, ranking_results: Dict[str, Any]) -> str:
        """Add comprehensive model ranking section to HTML report"""
        
        if not ranking_results or 'rankings' not in ranking_results:
            return "<h2>🏆 Model Rankings</h2><p>No ranking data available</p>"
        
        html = []
        html.append('<div class="ranking-section">')
        html.append('<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB;">🏆 Comprehensive Model Rankings</h2>')
        
        # Add executive summary
        top_model = ranking_results['rankings'][0]
        html.append('<div class="executive-summary" style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 15px 0;">')
        html.append('<h3 style="color: #1B4F72;">Executive Summary</h3>')
        html.append(f'<p><strong>Recommended Model:</strong> {top_model["model"]} (Score: {top_model["overall_score"]:.3f})</p>')
        html.append(f'<p><strong>Use Case Priority:</strong> {ranking_results["use_case_priority"].title()}</p>')
        html.append(f'<p><strong>Summary:</strong> {top_model["summary"]}</p>')
        html.append('</div>')
        
        # Add methodology explanation
        methodology = ranking_results.get('methodology', {})
        weights = methodology.get('criteria_weights', {})
        
        html.append('<h3 style="color: #2E86AB;">📋 Ranking Methodology</h3>')
        html.append('<div class="methodology" style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 15px 0;">')
        html.append('<p>Models are evaluated using a weighted multi-criteria scoring system:</p>')
        html.append('<ul>')
        for criterion, weight in weights.items():
            description = methodology.get('scoring_details', {}).get(criterion, '')
            html.append(f'<li><strong>{criterion.title()}</strong> ({weight:.1%}): {description}</li>')
        html.append('</ul>')
        html.append('<p><em>All scores are normalized to a 0-1 scale where 1.0 represents optimal performance.</em></p>')
        html.append('</div>')
        
        # Add comprehensive rankings table
        rankings = ranking_results['rankings']
        html.append('<h3 style="color: #2E86AB;">📊 Detailed Model Rankings</h3>')
        
        html.append('<div class="table-responsive">')
        html.append('<table class="ranking-table" style="width: 100%; border-collapse: collapse; margin: 15px 0;">')
        
        # Table headers
        headers = ['Rank', 'Model', 'Overall', 'Performance', 'Stability', 'Interpretability', 'Efficiency', 'Robustness', 'Summary']
        html.append('<thead style="background: #2E86AB; color: white;">')
        html.append('<tr>')
        for header in headers:
            html.append(f'<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">{header}</th>')
        html.append('</tr>')
        html.append('</thead>')
        
        # Table body
        html.append('<tbody>')
        for rank_data in rankings:
            # Highlight top model
            row_class = 'style="background: #d4edda; font-weight: bold;"' if rank_data['rank'] == 1 else ''
            html.append(f'<tr {row_class}>')
            
            # Rank with medal emoji
            rank_display = '🥇' if rank_data['rank'] == 1 else '🥈' if rank_data['rank'] == 2 else '🥉' if rank_data['rank'] == 3 else f"#{rank_data['rank']}"
            html.append(f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd;">{rank_display}</td>')
            
            html.append(f'<td style="padding: 10px; border: 1px solid #ddd;"><strong>{rank_data["model"]}</strong></td>')
            
            # Score cells with color coding
            score_style = 'padding: 10px; text-align: center; border: 1px solid #ddd;'
            html.append(f'<td style="{score_style} {self._get_score_color_style(rank_data["overall_score"])}">{rank_data["overall_score"]:.3f}</td>')
            html.append(f'<td style="{score_style} {self._get_score_color_style(rank_data["performance_score"])}">{rank_data["performance_score"]:.3f}</td>')
            html.append(f'<td style="{score_style} {self._get_score_color_style(rank_data["stability_score"])}">{rank_data["stability_score"]:.3f}</td>')
            html.append(f'<td style="{score_style} {self._get_score_color_style(rank_data["interpretability_score"])}">{rank_data["interpretability_score"]:.3f}</td>')
            html.append(f'<td style="{score_style} {self._get_score_color_style(rank_data["efficiency_score"])}">{rank_data["efficiency_score"]:.3f}</td>')
            html.append(f'<td style="{score_style} {self._get_score_color_style(rank_data["robustness_score"])}">{rank_data["robustness_score"]:.3f}</td>')
            
            html.append(f'<td style="padding: 10px; border: 1px solid #ddd; max-width: 200px;">{rank_data["summary"]}</td>')
            html.append('</tr>')
        
        html.append('</tbody>')
        html.append('</table>')
        html.append('</div>')
        
        # Add detailed model analysis
        html.append('<h3 style="color: #2E86AB;">🔍 Model Analysis Details</h3>')
        
        for model_data in rankings[:3]:  # Show top 3 models
            html.append('<div class="model-analysis" style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2E86AB;">')
            html.append(f'<h4 style="color: #1B4F72; margin-top: 0;">#{model_data["rank"]} - {model_data["model"]}</h4>')
            
            if model_data['key_strengths']:
                html.append('<p><strong>✅ Key Strengths:</strong></p>')
                html.append('<ul>')
                for strength in model_data['key_strengths']:
                    html.append(f'<li>{strength}</li>')
                html.append('</ul>')
            
            if model_data['main_weakness']:
                html.append(f'<p><strong>⚠️ Main Weakness:</strong> {model_data["main_weakness"]}</p>')
            
            if model_data['top_recommendation']:
                html.append(f'<p><strong>💡 Recommendation:</strong> {model_data["top_recommendation"]}</p>')
            
            html.append('</div>')
        
        # Add overall recommendations
        recommendations = ranking_results.get('recommendations', [])
        if recommendations:
            html.append('<h3 style="color: #2E86AB;">💡 Overall Recommendations</h3>')
            html.append('<div class="recommendations" style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin: 15px 0;">')
            html.append('<ol>')
            for rec in recommendations:
                html.append(f'<li>{rec}</li>')
            html.append('</ol>')
            html.append('</div>')
        
        # Add context about SHAP stability (addressing your original concern)
        html.append('<h3 style="color: #2E86AB;">🔬 Interpretability & SHAP Stability Notes</h3>')
        html.append('<div class="shap-notes" style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 15px 0;">')
        html.append('<p>This ranking system specifically addresses model interpretability reliability:</p>')
        html.append('<ul>')
        html.append('<li><strong>SHAP Validation:</strong> Models are scored based on SHAP analysis availability and mathematical validity</li>')
        html.append('<li><strong>Feature Importance Stability:</strong> Higher scores given to models with consistent feature attribution patterns</li>')
        html.append('<li><strong>Hyperparameter Sensitivity:</strong> Models penalized if feature importance changes dramatically with parameter tuning</li>')
        html.append('<li><strong>Model Transparency:</strong> Inherently interpretable models (linear, tree-based) receive bonus points</li>')
        html.append('</ul>')
        html.append('<p><em>For your Bagging model SHAP direction changes: This ranking system would flag such instability in the interpretability score, helping you identify more reliable models for feature analysis.</em></p>')
        html.append('</div>')
        
        html.append('</div>')
        
        return '\n'.join(html)

    def _get_score_color_style(self, score: float) -> str:
        """Return CSS style for score color coding"""
        if score >= 0.8:
            return 'background-color: #d4edda; color: #155724;'  # Green
        elif score >= 0.6:
            return 'background-color: #fff3cd; color: #856404;'  # Yellow
        elif score >= 0.4:
            return 'background-color: #ffeaa7; color: #6c5ce7;'  # Orange
        else:
            return 'background-color: #f8d7da; color: #721c24;'  # Red

    def _add_ranking_to_existing_report(self, sections: Dict[str, Any]) -> str:
        """Integrate ranking results into existing report structure"""
        
        ranking_results = sections.get('model_ranking')
        if not ranking_results:
            return ""
        
        # This method integrates with your existing report generation
        # Call this from your main generate_report method after other sections
        
        ranking_html = self._add_comprehensive_ranking_section(ranking_results)
        
        # Add some styling specific to integration
        integration_style = """
        <style>
        .ranking-section {
            margin: 20px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        .ranking-table th {
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9em;
        }
        .ranking-table td {
            font-size: 0.9em;
        }
        .executive-summary {
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .model-analysis {
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        </style>
        """
        
        return integration_style + ranking_html

    # ALSO MODIFY YOUR EXISTING generate_report METHOD
    # Find your generate_report method and add this after the existing sections:

    def generate_report_with_ranking(self, sections: Dict[str, Any], 
                                include_ranking: bool = True) -> str:
        """
        Enhanced version of generate_report that includes ranking section
        This can replace or supplement your existing generate_report method
        """
        
        html_sections = []
        
        # Add standard sections (your existing code)
        # ... your existing sections ...
        
        # Add ranking section if available and requested
        if include_ranking and 'model_ranking' in sections:
            ranking_html = self._add_ranking_to_existing_report(sections)
            if ranking_html:
                html_sections.append(ranking_html)
        
        # Combine all sections
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>AquaVista Analysis Report with Comprehensive Rankings</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2E86AB; }}
                .section {{ margin: 20px 0; padding: 15px; }}
                .highlight {{ background: #f0f8ff; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>AquaVista Water Quality Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {''.join(html_sections)}
        </body>
        </html>
        """
        
        return full_html
"""
Export Manager Module for AquaVista v6.0
=======================================
Handles model export, report generation, and deployment package creation.
"""

import joblib
import pickle
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import zipfile
import warnings
import logging
import pandas as pd
import numpy as np

# Import custom modules
from modules.config import Config
from modules.logging_config import get_logger, log_function_call

# Optional imports for different export formats
try:
    import onnx
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from sklearn2pmml import sklearn2pmml, PMMLPipeline
    PMML_AVAILABLE = True
except ImportError:
    PMML_AVAILABLE = False

try:
    import pdfkit
    from weasyprint import HTML
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class ExportManager:
    """Manages export of models, reports, and deployment packages"""
    
    def __init__(self, config: Config):
        self.config = config
        self.export_dir = config.export.output_dir
        self._ensure_export_directories()
    
    def _ensure_export_directories(self):
        """Ensure export directories exist"""
        directories = ['models', 'reports', 'deployments', 'data']
        for dir_name in directories:
            (self.export_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    @log_function_call(log_time=True)
    def export_models(self, training_results: Dict[str, Any],
                     format: str = 'joblib',
                     include_preprocessing: bool = True,
                     models_to_export: Optional[List[str]] = None) -> Path:
        """Export trained models
        
        Args:
            training_results: Training results dictionary
            format: Export format ('joblib', 'pickle', 'onnx', 'pmml')
            include_preprocessing: Include preprocessing pipeline
            models_to_export: Specific models to export (None = all)
            
        Returns:
            Path to export directory
        """
        # Create export directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = self.export_dir / 'models' / f'export_{timestamp}'
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Determine which models to export
        if models_to_export is None:
            models_to_export = list(training_results['models'].keys())
        
        exported_models = []
        
        for model_name in models_to_export:
            if model_name not in training_results['models']:
                logger.warning(f"Model '{model_name}' not found in results")
                continue
            
            model_data = training_results['models'][model_name]
            
            try:
                # Export based on format
                if format == 'joblib':
                    model_path = self._export_joblib(model_data, model_name, export_path)
                elif format == 'pickle':
                    model_path = self._export_pickle(model_data, model_name, export_path)
                elif format == 'onnx':
                    model_path = self._export_onnx(model_data, model_name, export_path)
                elif format == 'pmml':
                    model_path = self._export_pmml(model_data, model_name, export_path)
                else:
                    logger.error(f"Unsupported export format: {format}")
                    continue
                
                exported_models.append({
                    'model_name': model_name,
                    'file_path': str(model_path),
                    'format': format,
                    'metrics': model_data['test_scores']
                })
                
                logger.info(f"Exported {model_name} to {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to export {model_name}: {str(e)}")
        
        # Export metadata
        metadata = {
            'export_info': {
                'timestamp': timestamp,
                'format': format,
                'aquavista_version': '6.0.0',
                'include_preprocessing': include_preprocessing
            },
            'task_type': training_results['task_type'],
            'models': exported_models,
            'best_model': training_results['best_model'],
            'feature_names': training_results.get('feature_names', [])
        }
        
        metadata_path = export_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Export preprocessing pipeline if requested
        if include_preprocessing and 'preprocessing' in training_results:
            prep_path = export_path / 'preprocessing.joblib'
            joblib.dump(training_results['preprocessing'], prep_path)
        
        logger.info(f"Models exported to {export_path}")
        
        return export_path
    
    def _export_joblib(self, model_data: Dict[str, Any], model_name: str,
                      export_path: Path) -> Path:
        """Export model using joblib"""
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        file_path = export_path / f'{safe_name}.joblib'
        
        export_data = {
            'model': model_data['model'],
            'model_name': model_name,
            'feature_names': model_data.get('feature_names', []),
            'metrics': model_data['test_scores'],
            'training_time': model_data.get('training_time', 0),
            'best_params': model_data.get('best_params', {})
        }
        
        if self.config.export.compression == 'gzip':
            joblib.dump(export_data, file_path, compress=('gzip', 3))
        else:
            joblib.dump(export_data, file_path)
        
        return file_path
    
    def _export_pickle(self, model_data: Dict[str, Any], model_name: str,
                      export_path: Path) -> Path:
        """Export model using pickle"""
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        file_path = export_path / f'{safe_name}.pkl'
        
        export_data = {
            'model': model_data['model'],
            'model_name': model_name,
            'metrics': model_data['test_scores']
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(export_data, f)
        
        return file_path
    
    def _export_onnx(self, model_data: Dict[str, Any], model_name: str,
                    export_path: Path) -> Path:
        """Export model to ONNX format"""
        if not ONNX_AVAILABLE:
            raise ValueError("ONNX export requires onnx and skl2onnx packages")
        
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        file_path = export_path / f'{safe_name}.onnx'
        
        model = model_data['model']
        n_features = len(model_data.get('feature_names', []))
        
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert to ONNX
        try:
            onx = convert_sklearn(model, initial_types=initial_type)
            
            # Save
            with open(file_path, "wb") as f:
                f.write(onx.SerializeToString())
            
            logger.info(f"Exported {model_name} to ONNX format")
            
        except Exception as e:
            logger.error(f"ONNX export failed for {model_name}: {str(e)}")
            raise
        
        return file_path
    
    def _export_pmml(self, model_data: Dict[str, Any], model_name: str,
                    export_path: Path) -> Path:
        """Export model to PMML format"""
        if not PMML_AVAILABLE:
            raise ValueError("PMML export requires sklearn2pmml package")
        
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        file_path = export_path / f'{safe_name}.pmml'
        
        # PMML export requires pipeline
        # This is a simplified version - full implementation would need proper pipeline
        logger.warning("PMML export not fully implemented")
        
        return file_path
    
    @log_function_call(log_time=True)
    def generate_report(self, training_results: Dict[str, Any],
                       processed_data: Dict[str, Any],
                       format: str = 'html',
                       include_visualizations: bool = True) -> Path:
        """Generate comprehensive report
        
        Args:
            training_results: Training results
            processed_data: Processed data info
            format: Report format ('html', 'pdf', 'markdown', 'latex')
            include_visualizations: Include plots in report
            
        Returns:
            Path to generated report
        """
        # Create report directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.export_dir / 'reports' / f'report_{timestamp}'
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Generate report content
        if format == 'html':
            report_file = self._generate_html_report(
                training_results, processed_data, report_path, include_visualizations
            )
        elif format == 'pdf':
            report_file = self._generate_pdf_report(
                training_results, processed_data, report_path, include_visualizations
            )
        elif format == 'markdown':
            report_file = self._generate_markdown_report(
                training_results, processed_data, report_path
            )
        elif format == 'latex':
            report_file = self._generate_latex_report(
                training_results, processed_data, report_path
            )
        else:
            raise ValueError(f"Unsupported report format: {format}")
        
        logger.info(f"Report generated: {report_file}")
        
        return report_file
    
    def _generate_html_report(self, training_results: Dict[str, Any],
                            processed_data: Dict[str, Any],
                            report_path: Path,
                            include_visualizations: bool) -> Path:
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AquaVista ML Report</title>
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
        .best-model {{
            background-color: #ffffcc;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌊 AquaVista Machine Learning Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>[▊] Dataset Information</h2>
        <div class="metric">
            <strong>Training Samples:</strong> {processed_data['n_train']:,}
        </div>
        <div class="metric">
            <strong>Test Samples:</strong> {processed_data['n_test']:,}
        </div>
        <div class="metric">
            <strong>Features:</strong> {processed_data['n_features']}
        </div>
        <div class="metric">
            <strong>Task Type:</strong> {processed_data['task_type'].title()}
        </div>
        
        <h2>🏆 Best Model</h2>
        <p><strong>{training_results['best_model']}</strong> achieved the best performance 
        with {training_results['primary_metric']} = {training_results['best_score']:.4f}</p>
        
        <h2>[▲] Model Performance Summary</h2>
        {self._create_performance_table(training_results)}
        
        <h2>⚙️ Training Configuration</h2>
        <ul>
            <li>Cross-validation folds: {training_results.get('cv_folds', 'N/A')}</li>
            <li>Total training time: {training_results['total_time']:.2f} seconds</li>
            <li>Number of models trained: {len(training_results['models'])}</li>
        </ul>
"""
        
        if include_visualizations:
            html_content += """
        <h2>[▊] Visualizations</h2>
        <p>See attached visualization files in the report directory.</p>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Save HTML file
        report_file = report_path / 'report.html'
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        # Save additional data
        self._save_report_data(training_results, processed_data, report_path)
        
        return report_file
    
    def _create_performance_table(self, training_results: Dict[str, Any]) -> str:
        """Create HTML performance table"""
        html = "<table>\n<tr><th>Model</th><th>Training Time (s)</th>"
        
        # Add metric columns
        metrics = list(next(iter(training_results['models'].values()))['test_scores'].keys())
        for metric in metrics:
            html += f"<th>{metric.replace('_', ' ').title()}</th>"
        html += "</tr>\n"
        
        # Add model rows
        best_model = training_results['best_model']
        for model_name, model_data in training_results['models'].items():
            row_class = 'best-model' if model_name == best_model else ''
            html += f'<tr class="{row_class}">'
            html += f"<td>{model_name}</td>"
            html += f"<td>{model_data['training_time']:.2f}</td>"
            
            for metric in metrics:
                value = model_data['test_scores'][metric]
                html += f"<td>{value:.4f}</td>"
            
            html += "</tr>\n"
        
        html += "</table>"
        return html
    
    def _generate_pdf_report(self, training_results: Dict[str, Any],
                           processed_data: Dict[str, Any],
                           report_path: Path,
                           include_visualizations: bool) -> Path:
        """Generate PDF report"""
        if not PDF_AVAILABLE:
            logger.warning("PDF generation not available, generating HTML instead")
            return self._generate_html_report(
                training_results, processed_data, report_path, include_visualizations
            )
        
        # First generate HTML
        html_file = self._generate_html_report(
            training_results, processed_data, report_path, include_visualizations
        )
        
        # Convert to PDF
        pdf_file = report_path / 'report.pdf'
        
        try:
            HTML(filename=str(html_file)).write_pdf(str(pdf_file))
            logger.info(f"PDF report generated: {pdf_file}")
            return pdf_file
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            return html_file
    
    def _generate_markdown_report(self, training_results: Dict[str, Any],
                                processed_data: Dict[str, Any],
                                report_path: Path) -> Path:
        """Generate Markdown report"""
        md_content = f"""# 🌊 AquaVista Machine Learning Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## [▊] Dataset Information

- **Training Samples:** {processed_data['n_train']:,}
- **Test Samples:** {processed_data['n_test']:,}
- **Features:** {processed_data['n_features']}
- **Task Type:** {processed_data['task_type'].title()}

## 🏆 Best Model

**{training_results['best_model']}** achieved the best performance with {training_results['primary_metric']} = {training_results['best_score']:.4f}

## [▲] Model Performance Summary

| Model | Training Time (s) | {' | '.join([m.replace('_', ' ').title() for m in list(next(iter(training_results['models'].values()))['test_scores'].keys())])} |
|-------|------------------|{'-|' * len(list(next(iter(training_results['models'].values()))['test_scores'].keys()))}
"""
        
        # Add model rows
        for model_name, model_data in training_results['models'].items():
            row = f"| {model_name} | {model_data['training_time']:.2f} |"
            for metric, value in model_data['test_scores'].items():
                row += f" {value:.4f} |"
            md_content += row + "\n"
        
        md_content += f"""
## ⚙️ Training Configuration

- Cross-validation folds: {training_results.get('cv_folds', 'N/A')}
- Total training time: {training_results['total_time']:.2f} seconds
- Number of models trained: {len(training_results['models'])}

## 📋 Feature Information

Total features used: {processed_data['n_features']}
"""
        
        # Save Markdown file
        report_file = report_path / 'report.md'
        with open(report_file, 'w') as f:
            f.write(md_content)
        
        return report_file
    
    def _generate_latex_report(self, training_results: Dict[str, Any],
                             processed_data: Dict[str, Any],
                             report_path: Path) -> Path:
        """Generate LaTeX report"""
        latex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}

\title{AquaVista Machine Learning Report}
\author{AquaVista v6.0}
\date{""" + datetime.now().strftime('%B %d, %Y') + r"""}

\begin{document}
\maketitle

\section{Dataset Information}
\begin{itemize}
    \item Training Samples: """ + f"{processed_data['n_train']:,}" + r"""
    \item Test Samples: """ + f"{processed_data['n_test']:,}" + r"""
    \item Features: """ + str(processed_data['n_features']) + r"""
    \item Task Type: """ + processed_data['task_type'].title() + r"""
\end{itemize}

\section{Best Model}
\textbf{""" + training_results['best_model'] + r"""} achieved the best performance with """ + training_results['primary_metric'] + r""" = """ + f"{training_results['best_score']:.4f}" + r"""

\section{Model Performance Summary}
\begin{table}[h]
\centering
\begin{tabular}{l""" + 'r' * (1 + len(list(next(iter(training_results['models'].values()))['test_scores'].keys()))) + r"""}
\toprule
Model & Training Time (s) & """ + ' & '.join([m.replace('_', ' ').title() for m in list(next(iter(training_results['models'].values()))['test_scores'].keys())]) + r""" \\
\midrule
"""
        
        # Add model rows
        for model_name, model_data in training_results['models'].items():
            row = f"{model_name} & {model_data['training_time']:.2f}"
            for metric, value in model_data['test_scores'].items():
                row += f" & {value:.4f}"
            latex_content += row + r" \\" + "\n"
        
        latex_content += r"""\bottomrule
\end{tabular}
\end{table}

\end{document}
"""
        
        # Save LaTeX file
        report_file = report_path / 'report.tex'
        with open(report_file, 'w') as f:
            f.write(latex_content)
        
        return report_file
    
    def _save_report_data(self, training_results: Dict[str, Any],
                         processed_data: Dict[str, Any],
                         report_path: Path):
        """Save additional report data"""
        # Save detailed results
        results_file = report_path / 'detailed_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'training_results': {
                    k: v for k, v in training_results.items()
                    if k not in ['models']  # Exclude model objects
                },
                'model_metrics': {
                    name: data['test_scores']
                    for name, data in training_results['models'].items()
                },
                'processed_data_info': {
                    k: v for k, v in processed_data.items()
                    if k not in ['X_train', 'X_test', 'y_train', 'y_test']
                }
            }, f, indent=2, default=str)
        
        # Save performance summary as CSV
        perf_data = []
        for model_name, model_data in training_results['models'].items():
            row = {'Model': model_name, 'Training_Time': model_data['training_time']}
            row.update(model_data['test_scores'])
            perf_data.append(row)
        
        perf_df = pd.DataFrame(perf_data)
        perf_df.to_csv(report_path / 'performance_summary.csv', index=False)
    
    @log_function_call(log_time=True)
    def create_deployment_package(self, model_data: Dict[str, Any],
                                deployment_target: str = 'REST API') -> Path:
        """Create deployment package for model
        
        Args:
            model_data: Model data dictionary
            deployment_target: Target deployment ('REST API', 'Docker Container', 
                             'Cloud Function', 'Edge Device')
            
        Returns:
            Path to deployment package
        """
        # Create deployment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        deploy_path = self.export_dir / 'deployments' / f'deploy_{timestamp}'
        deploy_path.mkdir(parents=True, exist_ok=True)
        
        # Export model
        model_file = deploy_path / 'model.joblib'
        joblib.dump(model_data['model'], model_file)
        
        # Create deployment based on target
        if deployment_target == 'REST API':
            self._create_rest_api_deployment(model_data, deploy_path)
        elif deployment_target == 'Docker Container':
            self._create_docker_deployment(model_data, deploy_path)
        elif deployment_target == 'Cloud Function':
            self._create_cloud_function_deployment(model_data, deploy_path)
        elif deployment_target == 'Edge Device':
            self._create_edge_deployment(model_data, deploy_path)
        else:
            raise ValueError(f"Unsupported deployment target: {deployment_target}")
        
        # Create ZIP package
        zip_path = deploy_path.with_suffix('.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in deploy_path.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(deploy_path))
        
        logger.info(f"Deployment package created: {zip_path}")
        
        return zip_path
    
    def _create_rest_api_deployment(self, model_data: Dict[str, Any], deploy_path: Path):
        """Create REST API deployment"""
        # Create Flask app
        app_content = '''"""
AquaVista Model REST API
Generated by AquaVista v6.0
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('model.joblib')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': '""" + model_data.get('model_name', 'AquaVista Model') + """'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        if isinstance(data, dict):
            # Single prediction
            df = pd.DataFrame([data])
        else:
            # Batch prediction
            df = pd.DataFrame(data)
        
        # Make prediction
        predictions = model.predict(df)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df).tolist()
            response = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities
            }
        else:
            response = {'predictions': predictions.tolist()}
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
        
        # Save app file
        with open(deploy_path / 'app.py', 'w') as f:
            f.write(app_content)
        
        # Create requirements file
        requirements = """flask==2.3.2
joblib==1.3.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
"""
        
        with open(deploy_path / 'requirements.txt', 'w') as f:
            f.write(requirements)
        
        # Create README
        readme = f"""# AquaVista Model API

## Model Information
- Model: {model_data.get('model_name', 'Unknown')}
- Task Type: {model_data.get('task_type', 'Unknown')}
- Features: {len(model_data.get('feature_names', []))}

## Installation
```bash
pip install -r requirements.txt
```

## Running the API
```bash
python app.py
```

## API Endpoints

### Health Check
```
GET /health
```

### Prediction
```
POST /predict
Content-Type: application/json

# Single prediction
{{"feature1": value1, "feature2": value2, ...}}

# Batch prediction
[
    {{"feature1": value1, "feature2": value2, ...}},
    {{"feature1": value1, "feature2": value2, ...}}
]
```

## Features Expected
{json.dumps(model_data.get('feature_names', []), indent=2)}
"""
        
        with open(deploy_path / 'README.md', 'w') as f:
            f.write(readme)
    
    def _create_docker_deployment(self, model_data: Dict[str, Any], deploy_path: Path):
        """Create Docker deployment"""
        # First create REST API files
        self._create_rest_api_deployment(model_data, deploy_path)
        
        # Create Dockerfile
        dockerfile = """FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
"""
        
        with open(deploy_path / 'Dockerfile', 'w') as f:
            f.write(dockerfile)
        
        # Create docker-compose file
        compose = """version: '3.8'

services:
  aquavista-model:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
"""
        
        with open(deploy_path / 'docker-compose.yml', 'w') as f:
            f.write(compose)
    
    def _create_cloud_function_deployment(self, model_data: Dict[str, Any], deploy_path: Path):
        """Create cloud function deployment (AWS Lambda style)"""
        # Create Lambda handler
        handler_content = '''"""
AquaVista Model Lambda Handler
Generated by AquaVista v6.0
"""

import json
import joblib
import numpy as np
import pandas as pd

# Load model at cold start
model = joblib.load('model.joblib')

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        # Parse input
        body = json.loads(event.get('body', '{}'))
        
        # Convert to DataFrame
        if isinstance(body, dict):
            df = pd.DataFrame([body])
        else:
            df = pd.DataFrame(body)
        
        # Make prediction
        predictions = model.predict(df)
        
        # Build response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'predictions': predictions.tolist()
            })
        }
        
        return response
        
    except Exception as e:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
'''
        
        with open(deploy_path / 'lambda_function.py', 'w') as f:
            f.write(handler_content)
        
        # Create serverless.yml for deployment
        serverless = f"""service: aquavista-model

provider:
  name: aws
  runtime: python3.9
  stage: prod
  region: us-east-1

functions:
  predict:
    handler: lambda_function.lambda_handler
    events:
      - http:
          path: predict
          method: post
          cors: true

package:
  patterns:
    - '!./**'
    - lambda_function.py
    - model.joblib
    - requirements.txt
"""
        
        with open(deploy_path / 'serverless.yml', 'w') as f:
            f.write(serverless)
    
    def _create_edge_deployment(self, model_data: Dict[str, Any], deploy_path: Path):
        """Create edge device deployment"""
        # Create lightweight inference script
        inference_content = '''"""
AquaVista Model Edge Inference
Generated by AquaVista v6.0
Optimized for edge devices with limited resources
"""

import joblib
import numpy as np
import json
import sys

# Load model once
print("Loading model...")
model = joblib.load('model.joblib')
print("Model loaded successfully")

def predict(features):
    """Make prediction with error handling"""
    try:
        # Ensure correct shape
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return {
            'success': True,
            'prediction': float(prediction)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == '__main__':
    # Example usage - read from command line
    if len(sys.argv) > 1:
        try:
            # Parse features from command line
            features = json.loads(sys.argv[1])
            result = predict(features)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({'success': False, 'error': str(e)}))
    else:
        print("Usage: python inference.py '[feature1, feature2, ...]'")
'''
        
        with open(deploy_path / 'inference.py', 'w') as f:
            f.write(inference_content)
        
        # Create minimal requirements
        requirements = """joblib==1.3.0
numpy==1.24.0
scikit-learn==1.3.0
"""
        
        with open(deploy_path / 'requirements_edge.txt', 'w') as f:
            f.write(requirements)
        
        # Create optimization script
        optimize_script = '''"""
Optimize model for edge deployment
"""

import joblib
import pickle

# Load model
model = joblib.load('model.joblib')

# Save with high compression
with open('model_compressed.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Model compressed for edge deployment")
'''
        
        with open(deploy_path / 'optimize.py', 'w') as f:
            f.write(optimize_script)
    
    def export_predictions(self, predictions: np.ndarray, 
                         output_path: Optional[Path] = None,
                         format: str = 'csv') -> Path:
        """Export predictions to file
        
        Args:
            predictions: Model predictions
            output_path: Output file path
            format: Export format ('csv', 'json', 'excel')
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.export_dir / 'data' / f'predictions_{timestamp}.{format}'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame({
            'prediction': predictions,
            'timestamp': datetime.now()
        })
        
        # Export based on format
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', date_format='iso')
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Predictions exported to {output_path}")
        
        return output_path
"""
Performance Monitoring Module for AquaVista v6.0
===============================================
Monitors system resources, tracks performance metrics, and provides optimization recommendations.
"""

import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import time
import threading
import queue
from datetime import datetime, timedelta
from collections import deque, defaultdict
import warnings
import logging
import json
from pathlib import Path

# Import custom modules
from modules.config import Config
from modules.logging_config import get_performance_logger, log_function_call

logger = get_performance_logger()
warnings.filterwarnings('ignore')


class PerformanceMonitor:
    """Monitors and tracks system performance metrics"""
    
    def __init__(self, config: Config):
        self.config = config
        self.start_time = time.time()
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = deque(maxlen=100)
        self.model_training_times = {}
        self.peak_memory = 0
        self.total_models_trained = 0
        
        # Performance thresholds
        self.thresholds = {
            'memory_warning': 70,  # %
            'memory_critical': 85,  # %
            'cpu_warning': 80,     # %
            'cpu_critical': 95,    # %
            'disk_warning': 90,    # %
            'response_time_warning': 5.0,  # seconds
        }
        
        # Start monitoring thread if enabled
        self.monitoring_enabled = True
        self.monitoring_thread = None
        self.monitoring_queue = queue.Queue()
        
        if self.config.features.memory_guard:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Collect metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                timestamp = datetime.now()
                for key, value in metrics.items():
                    self.metrics_history[key].append((timestamp, value))
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Update peak memory
                if metrics['memory_percent'] > self.peak_memory:
                    self.peak_memory = metrics['memory_percent']
                
                # Sleep for monitoring interval
                time.sleep(5)  # 5 second intervals
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        metrics['cpu_count'] = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_used_gb'] = memory.used / (1024**3)
        metrics['memory_available_gb'] = memory.available / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics['disk_percent'] = disk.percent
        metrics['disk_free_gb'] = disk.free / (1024**3)
        
        # Process-specific metrics
        process = psutil.Process()
        metrics['process_memory_mb'] = process.memory_info().rss / (1024**2)
        metrics['process_cpu_percent'] = process.cpu_percent()
        
        # GPU metrics if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['gpu_temperature'] = gpu.temperature
        except:
            pass
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check metrics against thresholds and generate alerts"""
        timestamp = datetime.now()
        
        # Memory alerts
        if metrics['memory_percent'] > self.thresholds['memory_critical']:
            alert = {
                'timestamp': timestamp,
                'level': 'critical',
                'type': 'memory',
                'message': f"Critical memory usage: {metrics['memory_percent']:.1f}%",
                'value': metrics['memory_percent']
            }
            self.alerts.append(alert)
            logger.critical(alert['message'])
            
        elif metrics['memory_percent'] > self.thresholds['memory_warning']:
            alert = {
                'timestamp': timestamp,
                'level': 'warning',
                'type': 'memory',
                'message': f"High memory usage: {metrics['memory_percent']:.1f}%",
                'value': metrics['memory_percent']
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
        
        # CPU alerts
        if metrics['cpu_percent'] > self.thresholds['cpu_critical']:
            alert = {
                'timestamp': timestamp,
                'level': 'critical',
                'type': 'cpu',
                'message': f"Critical CPU usage: {metrics['cpu_percent']:.1f}%",
                'value': metrics['cpu_percent']
            }
            self.alerts.append(alert)
            logger.critical(alert['message'])
            
        elif metrics['cpu_percent'] > self.thresholds['cpu_warning']:
            alert = {
                'timestamp': timestamp,
                'level': 'warning',
                'type': 'cpu',
                'message': f"High CPU usage: {metrics['cpu_percent']:.1f}%",
                'value': metrics['cpu_percent']
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
        
        # Disk alerts
        if metrics['disk_percent'] > self.thresholds['disk_warning']:
            alert = {
                'timestamp': timestamp,
                'level': 'warning',
                'type': 'disk',
                'message': f"Low disk space: {metrics['disk_free_gb']:.1f} GB free",
                'value': metrics['disk_percent']
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
    
    @log_function_call
    def check_memory_availability(self, required_mb: float) -> bool:
        """Check if sufficient memory is available
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            True if memory is available
            
        Raises:
            MemoryError if insufficient memory
        """
        if not self.config.features.memory_guard:
            return True
        
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024**2)
        
        # Check against memory limit
        memory_limit_mb = memory.total * self.config.computation.memory_limit / 100 / (1024**2)
        current_usage_mb = memory.used / (1024**2)
        
        if current_usage_mb + required_mb > memory_limit_mb:
            raise MemoryError(
                f"Operation requires {required_mb:.1f} MB but would exceed "
                f"memory limit ({self.config.computation.memory_limit}% = {memory_limit_mb:.1f} MB)"
            )
        
        if required_mb > available_mb * 0.8:  # Keep 20% buffer
            raise MemoryError(
                f"Insufficient memory: {required_mb:.1f} MB required, "
                f"only {available_mb:.1f} MB available"
            )
        
        logger.info(f"Memory check passed: {required_mb:.1f} MB required, {available_mb:.1f} MB available")
        return True
    
    def get_system_health(self) -> Dict[str, Dict[str, Any]]:
        """Get current system health status
        
        Returns:
            Dictionary with health status for each component
        """
        metrics = self._collect_system_metrics()
        health_status = {}
        
        # CPU health
        cpu_percent = metrics['cpu_percent']
        if cpu_percent < self.thresholds['cpu_warning']:
            cpu_status = 'healthy'
            cpu_message = f"CPU usage normal ({cpu_percent:.1f}%)"
        elif cpu_percent < self.thresholds['cpu_critical']:
            cpu_status = 'warning'
            cpu_message = f"CPU usage high ({cpu_percent:.1f}%)"
        else:
            cpu_status = 'critical'
            cpu_message = f"CPU usage critical ({cpu_percent:.1f}%)"
        
        health_status['CPU'] = {
            'status': cpu_status,
            'message': cpu_message,
            'value': cpu_percent,
            'threshold_warning': self.thresholds['cpu_warning'],
            'threshold_critical': self.thresholds['cpu_critical']
        }
        
        # Memory health
        memory_percent = metrics['memory_percent']
        if memory_percent < self.thresholds['memory_warning']:
            memory_status = 'healthy'
            memory_message = f"Memory usage normal ({memory_percent:.1f}%)"
        elif memory_percent < self.thresholds['memory_critical']:
            memory_status = 'warning'
            memory_message = f"Memory usage high ({memory_percent:.1f}%)"
        else:
            memory_status = 'critical'
            memory_message = f"Memory usage critical ({memory_percent:.1f}%)"
        
        health_status['Memory'] = {
            'status': memory_status,
            'message': memory_message,
            'value': memory_percent,
            'available_gb': metrics['memory_available_gb'],
            'threshold_warning': self.thresholds['memory_warning'],
            'threshold_critical': self.thresholds['memory_critical']
        }
        
        # Disk health
        disk_percent = metrics['disk_percent']
        if disk_percent < self.thresholds['disk_warning']:
            disk_status = 'healthy'
            disk_message = f"Disk space adequate ({metrics['disk_free_gb']:.1f} GB free)"
        else:
            disk_status = 'warning'
            disk_message = f"Low disk space ({metrics['disk_free_gb']:.1f} GB free)"
        
        health_status['Disk'] = {
            'status': disk_status,
            'message': disk_message,
            'value': disk_percent,
            'free_gb': metrics['disk_free_gb'],
            'threshold_warning': self.thresholds['disk_warning']
        }
        
        # GPU health if available
        if 'gpu_memory_percent' in metrics:
            gpu_percent = metrics['gpu_memory_percent']
            gpu_temp = metrics.get('gpu_temperature', 0)
            
            if gpu_percent < 80 and gpu_temp < 80:
                gpu_status = 'healthy'
                gpu_message = f"GPU normal (Memory: {gpu_percent:.1f}%, Temp: {gpu_temp}°C)"
            else:
                gpu_status = 'warning'
                gpu_message = f"GPU stressed (Memory: {gpu_percent:.1f}%, Temp: {gpu_temp}°C)"
            
            health_status['GPU'] = {
                'status': gpu_status,
                'message': gpu_message,
                'memory_percent': gpu_percent,
                'temperature': gpu_temp,
                'utilization': metrics.get('gpu_utilization', 0)
            }
        
        return health_status
    
    def log_event(self, event_type: str, details: Optional[Dict[str, Any]] = None):
        """Log a performance-related event
        
        Args:
            event_type: Type of event
            details: Additional event details
        """
        timestamp = datetime.now()
        event = {
            'timestamp': timestamp,
            'type': event_type,
            'details': details or {}
        }
        
        # Store in appropriate history
        if event_type.startswith('model_'):
            if 'model_name' in details:
                model_name = details['model_name']
                if event_type == 'model_training_start':
                    self.model_training_times[model_name] = {'start': timestamp}
                elif event_type == 'model_training_end':
                    if model_name in self.model_training_times:
                        self.model_training_times[model_name]['end'] = timestamp
                        self.model_training_times[model_name]['duration'] = (
                            timestamp - self.model_training_times[model_name]['start']
                        ).total_seconds()
                    self.total_models_trained += 1
        
        logger.info(f"Performance event: {event_type}", extra=details)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics
        
        Returns:
            Dictionary with session statistics
        """
        current_time = time.time()
        runtime_seconds = current_time - self.start_time
        
        # Format runtime
        hours = int(runtime_seconds // 3600)
        minutes = int((runtime_seconds % 3600) // 60)
        seconds = int(runtime_seconds % 60)
        
        if hours > 0:
            runtime_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            runtime_str = f"{minutes}m {seconds}s"
        else:
            runtime_str = f"{seconds}s"
        
        stats = {
            'runtime': runtime_str,
            'runtime_seconds': runtime_seconds,
            'models_trained': self.total_models_trained,
            'peak_memory': self.peak_memory,
            'alerts_count': len(self.alerts),
            'critical_alerts': sum(1 for a in self.alerts if a['level'] == 'critical'),
            'warning_alerts': sum(1 for a in self.alerts if a['level'] == 'warning')
        }
        
        # Average training time
        if self.model_training_times:
            training_durations = [
                m.get('duration', 0) for m in self.model_training_times.values()
                if 'duration' in m
            ]
            if training_durations:
                stats['avg_training_time'] = np.mean(training_durations)
                stats['total_training_time'] = sum(training_durations)
        
        return stats
    
    def get_resource_usage_history(self, metric: str = 'memory_percent',
                                 duration_minutes: int = 30) -> pd.DataFrame:
        """Get resource usage history
        
        Args:
            metric: Metric to retrieve
            duration_minutes: Duration to look back
            
        Returns:
            DataFrame with timestamp and values
        """
        if metric not in self.metrics_history:
            return pd.DataFrame(columns=['timestamp', 'value'])
        
        # Get data from history
        history = self.metrics_history[metric]
        
        # Filter by duration
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        filtered_data = [(ts, val) for ts, val in history if ts >= cutoff_time]
        
        if not filtered_data:
            return pd.DataFrame(columns=['timestamp', 'value'])
        
        # Create DataFrame
        df = pd.DataFrame(filtered_data, columns=['timestamp', 'value'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Get optimization recommendations based on current performance
        
        Returns:
            List of recommendations
        """
        recommendations = []
        metrics = self._collect_system_metrics()
        
        # Memory recommendations
        if metrics['memory_percent'] > self.thresholds['memory_warning']:
            recommendations.append({
                'category': 'Memory',
                'priority': 'high',
                'title': 'Reduce memory usage',
                'description': 'Consider reducing batch size, using fewer models, or enabling memory-efficient options',
                'actions': [
                    'Set config.features["memory_guard"] = True',
                    'Reduce config.computation["chunk_size"]',
                    'Use fewer ensemble models',
                    'Enable garbage collection between operations'
                ]
            })
        
        # CPU recommendations
        if metrics['cpu_percent'] > self.thresholds['cpu_warning']:
            recommendations.append({
                'category': 'CPU',
                'priority': 'medium',
                'title': 'Optimize CPU usage',
                'description': 'CPU usage is high, consider reducing parallel operations',
                'actions': [
                    f'Reduce n_jobs from {self.config.computation.n_jobs} to {max(1, self.config.computation.n_jobs // 2)}',
                    'Use simpler models (e.g., Linear instead of ensemble)',
                    'Disable hyperparameter optimization'
                ]
            })
        
        # Model-specific recommendations
        if self.total_models_trained > 10:
            avg_time = self.get_session_stats().get('avg_training_time', 0)
            if avg_time > 60:  # More than 1 minute average
                recommendations.append({
                    'category': 'Training',
                    'priority': 'medium',
                    'title': 'Speed up model training',
                    'description': f'Average training time is {avg_time:.1f}s',
                    'actions': [
                        'Use "speed" performance mode',
                        'Reduce CV folds',
                        'Use quick hyperparameter tuning',
                        'Consider sampling data for exploration'
                    ]
                })
        
        # Feature recommendations
        if self.config.features.shap_analysis if hasattr(config.features, 'shap_analysis') else True and metrics['memory_percent'] > 60:
            recommendations.append({
                'category': 'Features',
                'priority': 'low',
                'title': 'Disable memory-intensive features',
                'description': 'Some features use significant memory',
                'actions': [
                    'Set config.features["shap_analysis"] = False',
                    'Set config.features["cv_visualization"] = False',
                    'Disable auto profiling for large datasets'
                ]
            })
        
        # Data size recommendations
        if hasattr(self, 'last_data_size') and self.last_data_size > 100000:
            recommendations.append({
                'category': 'Data',
                'priority': 'medium',
                'title': 'Consider data sampling',
                'description': 'Large dataset detected',
                'actions': [
                    'Use stratified sampling for initial exploration',
                    'Increase chunk_size for better performance',
                    'Consider dimensionality reduction (PCA)',
                    'Use efficient data formats (parquet)'
                ]
            })
        
        # Disk space recommendations
        if metrics['disk_percent'] > self.thresholds['disk_warning']:
            recommendations.append({
                'category': 'Disk',
                'priority': 'high',
                'title': 'Low disk space',
                'description': f'Only {metrics["disk_free_gb"]:.1f} GB free',
                'actions': [
                    'Clear old results in aquavista_results/',
                    'Disable result caching',
                    'Export models with compression',
                    'Remove temporary files'
                ]
            })
        
        return recommendations
    
    def create_performance_report(self) -> Dict[str, Any]:
        """Create comprehensive performance report
        
        Returns:
            Performance report dictionary
        """
        report = {
            'summary': self.get_session_stats(),
            'health': self.get_system_health(),
            'recommendations': self.get_optimization_recommendations(),
            'alerts': list(self.alerts)[-10:],  # Last 10 alerts
            'metrics': {}
        }
        
        # Add current metrics
        current_metrics = self._collect_system_metrics()
        report['current_metrics'] = current_metrics
        
        # Add metric trends
        for metric in ['cpu_percent', 'memory_percent']:
            history_df = self.get_resource_usage_history(metric, duration_minutes=60)
            if not history_df.empty:
                report['metrics'][metric] = {
                    'current': current_metrics.get(metric, 0),
                    'mean': history_df['value'].mean(),
                    'max': history_df['value'].max(),
                    'min': history_df['value'].min(),
                    'std': history_df['value'].std()
                }
        
        # Model training performance
        if self.model_training_times:
            model_perf = []
            for model_name, times in self.model_training_times.items():
                if 'duration' in times:
                    model_perf.append({
                        'model': model_name,
                        'duration': times['duration'],
                        'start': times['start'].isoformat(),
                        'end': times.get('end', datetime.now()).isoformat()
                    })
            report['model_performance'] = sorted(model_perf, key=lambda x: x['duration'], reverse=True)
        
        return report
    
    def save_performance_log(self, filepath: Optional[Path] = None):
        """Save performance log to file
        
        Args:
            filepath: Output file path
        """
        if filepath is None:
            filepath = Path("aquavista_results/logs/performance_log.json")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare log data
        log_data = {
            'session_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration': time.time() - self.start_time
            },
            'statistics': self.get_session_stats(),
            'alerts': [
                {
                    'timestamp': alert['timestamp'].isoformat(),
                    'level': alert['level'],
                    'type': alert['type'],
                    'message': alert['message']
                }
                for alert in self.alerts
            ],
            'model_training': self.model_training_times,
            'peak_memory': self.peak_memory,
            'final_metrics': self._collect_system_metrics()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Performance log saved to {filepath}")
    
    def cleanup(self):
        """Cleanup monitoring resources"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Save final log
        self.save_performance_log()
        
        logger.info("Performance monitor cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass
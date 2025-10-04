"""
Logging Configuration Module for AquaVista v6.0
==============================================
Provides comprehensive logging setup with structured logging,
rotation, and specialized loggers for different components.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from functools import wraps
import threading
import queue

# Try to import colorlog for colored console output
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

# Try to import python-json-logger for structured logging
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False


class AquaVistaFormatter(logging.Formatter):
    """Custom formatter for AquaVista logs"""
    
    def __init__(self, include_color: bool = True):
        self.include_color = include_color and COLORLOG_AVAILABLE
        
        if self.include_color:
            self.formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            self.formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    
    def format(self, record):
        return self.formatter.format(record)


class StructuredFormatter(logging.Formatter):
    """JSON structured formatter for machine-readable logs"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_obj['session_id'] = record.session_id
        if hasattr(record, 'model_name'):
            log_obj['model_name'] = record.model_name
        if hasattr(record, 'dataset_name'):
            log_obj['dataset_name'] = record.dataset_name
            
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)


class AsyncHandler(logging.Handler):
    """Asynchronous logging handler for performance"""
    
    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()
    
    def _worker(self):
        """Worker thread for processing log records"""
        while True:
            try:
                record = self.queue.get()
                if record is None:
                    break
                self.handler.emit(record)
            except Exception:
                import sys
                sys.stderr.write('Error in logging worker thread\n')
    
    def emit(self, record):
        """Add record to queue"""
        self.queue.put(record)
    
    def close(self):
        """Close the handler"""
        self.queue.put(None)
        self.thread.join()
        self.handler.close()
        super().close()


def setup_logging(log_level: str = 'INFO',
                 log_dir: Optional[Path] = None,
                 console_output: bool = True,
                 file_output: bool = True,
                 structured_logs: bool = False,
                 async_logging: bool = False) -> logging.Logger:
    """Setup logging configuration for AquaVista
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: aquavista_results/logs)
        console_output: Enable console logging
        file_output: Enable file logging
        structured_logs: Use JSON structured logging
        async_logging: Use asynchronous logging for better performance
        
    Returns:
        Logger instance
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("aquavista_results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if structured_logs and JSON_LOGGER_AVAILABLE:
            console_handler.setFormatter(jsonlogger.JsonFormatter())
        else:
            console_handler.setFormatter(AquaVistaFormatter(include_color=True))
        
        if async_logging:
            console_handler = AsyncHandler(console_handler)
            
        root_logger.addHandler(console_handler)
    
    # File handlers
    if file_output:
        # Main log file with rotation
        main_log_file = log_dir / f"aquavista_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if structured_logs:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(AquaVistaFormatter(include_color=False))
        
        if async_logging:
            file_handler = AsyncHandler(file_handler)
            
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(AquaVistaFormatter(include_color=False))
        
        if async_logging:
            error_handler = AsyncHandler(error_handler)
            
        root_logger.addHandler(error_handler)
    
    # Create specialized loggers
    create_specialized_loggers(log_level)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, Console: {console_output}, File: {file_output}")
    
    return logger


def create_specialized_loggers(log_level: str):
    """Create specialized loggers for different components"""
    
    # Model logger - for model training and evaluation
    model_logger = logging.getLogger('aquavista.models')
    model_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Data logger - for data processing
    data_logger = logging.getLogger('aquavista.data')
    data_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Performance logger - for performance metrics
    perf_logger = logging.getLogger('aquavista.performance')
    perf_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Audit logger - for audit trail
    audit_logger = logging.getLogger('aquavista.audit')
    audit_logger.setLevel(logging.INFO)  # Always INFO for audit


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance
    
    Args:
        name: Logger name (default: caller's module)
        
    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'aquavista')
    
    return logging.getLogger(name)


def get_model_logger() -> logging.Logger:
    """Get the model-specific logger"""
    return logging.getLogger('aquavista.models')


def get_data_logger() -> logging.Logger:
    """Get the data-specific logger"""
    return logging.getLogger('aquavista.data')


def get_performance_logger() -> logging.Logger:
    """Get the performance-specific logger"""
    return logging.getLogger('aquavista.performance')


def get_audit_logger() -> logging.Logger:
    """Get the audit logger"""
    return logging.getLogger('aquavista.audit')


def log_function_call(func: Optional[Callable] = None,
                     log_args: bool = True,
                     log_result: bool = False,
                     log_time: bool = True):
    """Decorator to log function calls
    
    Args:
        func: Function to decorate
        log_args: Log function arguments
        log_result: Log function result
        log_time: Log execution time
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            logger = get_logger(f.__module__)
            
            # Log function call
            msg = f"Calling {f.__name__}"
            if log_args:
                msg += f" with args={args}, kwargs={kwargs}"
            logger.debug(msg)
            
            # Execute function
            start_time = datetime.now()
            try:
                result = f(*args, **kwargs)
                
                # Log success
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                msg = f"{f.__name__} completed"
                if log_time:
                    msg += f" in {duration:.2f}s"
                if log_result:
                    msg += f" with result={result}"
                    
                logger.debug(msg)
                return result
                
            except Exception as e:
                # Log error
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.error(f"{f.__name__} failed after {duration:.2f}s: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


def create_audit_log(action: str,
                    user: Optional[str] = None,
                    details: Optional[Dict[str, Any]] = None,
                    status: str = 'success'):
    """Create an audit log entry
    
    Args:
        action: Action performed
        user: User who performed the action
        details: Additional details
        status: Status of the action (success, failure, warning)
    """
    audit_logger = get_audit_logger()
    
    audit_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'action': action,
        'user': user or 'system',
        'status': status,
        'details': details or {}
    }
    
    # Log as JSON
    audit_logger.info(json.dumps(audit_entry))


class LogContext:
    """Context manager for adding context to logs"""
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **factory_kwargs):
            record = self.old_factory(*args, **factory_kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def configure_external_loggers(level: str = 'WARNING'):
    """Configure logging levels for external libraries
    
    Args:
        level: Log level for external libraries
    """
    # Suppress verbose logs from external libraries
    external_loggers = [
        'urllib3',
        'requests',
        'matplotlib',
        'PIL',
        'tensorflow',
        'h5py',
        'numba',
        'werkzeug'
    ]
    
    for logger_name in external_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))


# Convenience functions for common logging patterns
def log_model_training(model_name: str, start: bool = True, **kwargs):
    """Log model training events
    
    Args:
        model_name: Name of the model
        start: True if starting training, False if completed
        **kwargs: Additional parameters to log
    """
    logger = get_model_logger()
    
    if start:
        logger.info(f"Starting training for {model_name}", extra={'model_name': model_name, **kwargs})
    else:
        logger.info(f"Completed training for {model_name}", extra={'model_name': model_name, **kwargs})


def log_data_processing(operation: str, dataset_name: str, **kwargs):
    """Log data processing operations
    
    Args:
        operation: Operation being performed
        dataset_name: Name of the dataset
        **kwargs: Additional parameters to log
    """
    logger = get_data_logger()
    logger.info(f"Data operation: {operation}", extra={'dataset_name': dataset_name, **kwargs})


def log_performance_metric(metric_name: str, value: float, **kwargs):
    """Log performance metrics
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        **kwargs: Additional context
    """
    logger = get_performance_logger()
    logger.info(f"Performance metric: {metric_name}={value:.4f}", extra={'metric': metric_name, 'value': value, **kwargs})


# Initialize external logger configuration on import
configure_external_loggers()
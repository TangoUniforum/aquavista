"""
Data Loader Module for AquaVista v6.0
====================================
Handles loading data from various file formats and sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import sqlite3
import warnings
import logging
from datetime import datetime

# Import custom modules
from modules.config import Config, DataLoadingError
from modules.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles data loading from various sources"""
    
    def __init__(self, config: Config):
        self.config = config
        self.supported_formats = {
            'csv': self._load_csv,
            'xlsx': self._load_excel,
            'xls': self._load_excel,
            'json': self._load_json,
            'parquet': self._load_parquet,
            'sqlite': self._load_sqlite,
            'db': self._load_sqlite
        }
        
    def load_data(self, file_path: Union[str, Path], 
                  file_type: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            file_path: Path to the data file
            file_type: Type of file (auto-detected if None)
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise DataLoadingError(f"File not found: {file_path}")
        
        # Auto-detect file type
        if file_type is None:
            file_type = file_path.suffix.lower().strip('.')
        
        # Check if format is supported
        if file_type not in self.supported_formats:
            raise DataLoadingError(
                f"Unsupported file format: {file_type}. "
                f"Supported formats: {list(self.supported_formats.keys())}"
            )
        
        # Load data
        try:
            logger.info(f"Loading {file_type} file: {file_path}")
            loader_func = self.supported_formats[file_type]
            df = loader_func(file_path, **kwargs)
            
            # Validate loaded data
            self._validate_loaded_data(df)
            
            logger.info(f"Successfully loaded data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise DataLoadingError(f"Failed to load {file_path}: {str(e)}")
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        # Default parameters
        params = {
            'encoding': 'utf-8',
            'low_memory': False,
            'na_values': ['NA', 'N/A', 'null', 'NULL', '']
        }
        params.update(kwargs)
        
        try:
            # Try loading with default encoding
            df = pd.read_csv(file_path, **params)
        except UnicodeDecodeError:
            # Try with different encodings
            encodings = ['latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    params['encoding'] = encoding
                    df = pd.read_csv(file_path, **params)
                    logger.info(f"Successfully loaded with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise DataLoadingError("Could not decode CSV with any common encoding")
        
        return df
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file"""
        # Check for sheet_name parameter
        sheet_name = kwargs.pop('sheet_name', 0)
        
        try:
            # Load Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            
            # If multiple sheets requested, combine them
            if isinstance(df, dict):
                # Combine all sheets
                combined_df = pd.DataFrame()
                for sheet, sheet_df in df.items():
                    sheet_df['_sheet_name'] = sheet
                    combined_df = pd.concat([combined_df, sheet_df], ignore_index=True)
                df = combined_df
                logger.info(f"Combined {len(df)} sheets from Excel file")
                
        except Exception as e:
            raise DataLoadingError(f"Error reading Excel file: {str(e)}")
        
        return df
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON file"""
        # Default orient
        orient = kwargs.pop('orient', 'records')
        
        try:
            # Try loading as DataFrame directly
            df = pd.read_json(file_path, orient=orient, **kwargs)
        except:
            # Try loading as raw JSON first
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try to determine structure
                    if all(isinstance(v, list) for v in data.values()):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.DataFrame([data])
                else:
                    raise DataLoadingError(f"Unsupported JSON structure: {type(data)}")
                    
            except Exception as e:
                raise DataLoadingError(f"Error parsing JSON file: {str(e)}")
        
        return df
    
    def _load_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file"""
        try:
            df = pd.read_parquet(file_path, **kwargs)
        except Exception as e:
            raise DataLoadingError(f"Error reading Parquet file: {str(e)}")
        
        return df
    
    def _load_sqlite(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from SQLite database"""
        # Get table name or query
        table_name = kwargs.pop('table_name', None)
        query = kwargs.pop('query', None)
        
        if not table_name and not query:
            raise DataLoadingError("Either 'table_name' or 'query' must be provided for SQLite")
        
        try:
            # Connect to database
            conn = sqlite3.connect(file_path)
            
            if query:
                df = pd.read_sql_query(query, conn, **kwargs)
            else:
                # If no query, load entire table
                df = pd.read_sql_table(table_name, conn, **kwargs)
            
            conn.close()
            
        except Exception as e:
            raise DataLoadingError(f"Error reading SQLite database: {str(e)}")
        
        return df
    
    def _validate_loaded_data(self, df: pd.DataFrame):
        """Validate loaded DataFrame"""
        # Check if empty
        if df.empty:
            raise DataLoadingError("Loaded dataframe is empty")
        
        # Check minimum size
        if len(df) < 10:
            logger.warning(f"Small dataset loaded: only {len(df)} rows")
        
        # Check for all null columns
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            logger.warning(f"Columns with all null values: {null_columns}")
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            raise DataLoadingError(f"Duplicate column names found: {duplicate_cols}")
    
    def load_from_url(self, url: str, file_type: Optional[str] = None, 
                     **kwargs) -> pd.DataFrame:
        """Load data from URL"""
        import requests
        import tempfile
        
        try:
            # Download file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type from URL if not provided
            if file_type is None:
                file_type = url.split('.')[-1].lower()
                if '?' in file_type:
                    file_type = file_type.split('?')[0]
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{file_type}', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = Path(tmp_file.name)
            
            # Load data
            df = self.load_data(tmp_path, file_type=file_type, **kwargs)
            
            # Clean up
            tmp_path.unlink()
            
            return df
            
        except Exception as e:
            raise DataLoadingError(f"Error loading data from URL: {str(e)}")
    
    def load_multiple_files(self, file_paths: List[Union[str, Path]], 
                          combine_method: str = 'concat',
                          **kwargs) -> pd.DataFrame:
        """Load and combine multiple files"""
        dataframes = []
        
        for file_path in file_paths:
            try:
                df = self.load_data(file_path, **kwargs)
                dataframes.append(df)
                logger.info(f"Loaded file {file_path}: {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                if not kwargs.get('skip_errors', False):
                    raise
        
        if not dataframes:
            raise DataLoadingError("No files were successfully loaded")
        
        # Combine dataframes
        if combine_method == 'concat':
            combined_df = pd.concat(dataframes, ignore_index=True)
        elif combine_method == 'merge':
            # Merge requires 'on' parameter
            merge_on = kwargs.get('merge_on')
            if not merge_on:
                raise DataLoadingError("'merge_on' parameter required for merge method")
            
            combined_df = dataframes[0]
            for df in dataframes[1:]:
                combined_df = combined_df.merge(df, on=merge_on, how=kwargs.get('how', 'inner'))
        else:
            raise DataLoadingError(f"Unknown combine method: {combine_method}")
        
        logger.info(f"Combined {len(dataframes)} files. Final shape: {combined_df.shape}")
        
        return combined_df
    
    def save_data(self, df: pd.DataFrame, file_path: Union[str, Path], 
                 file_type: Optional[str] = None, **kwargs):
        """Save DataFrame to file"""
        file_path = Path(file_path)
        
        # Auto-detect file type
        if file_type is None:
            file_type = file_path.suffix.lower().strip('.')
        
        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file type
        if file_type == 'csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif file_type in ['xlsx', 'xls']:
            df.to_excel(file_path, index=False, **kwargs)
        elif file_type == 'json':
            df.to_json(file_path, **kwargs)
        elif file_type == 'parquet':
            df.to_parquet(file_path, **kwargs)
        else:
            raise DataLoadingError(f"Unsupported save format: {file_type}")
        
        logger.info(f"Data saved to {file_path}")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataframe"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
        }
        
        return info
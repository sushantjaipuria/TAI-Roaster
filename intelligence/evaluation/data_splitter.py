"""
Data Splitter - Temporal Data Separation Module
Ensures strict temporal separation to prevent data leakage in model evaluation.

Requirements from PRD:
- Strict split logic with training window (start_date to 2024-06-30)
- Evaluation window (2024-07-01 to current)
- Data leakage prevention
- Validation methods to ensure no overlap
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataSplitter:
    """
    Handles temporal data splitting to prevent data leakage in evaluation.
    
    Features:
    - Strict temporal separation based on cutoff date
    - Data leakage validation
    - Time window creation for multiple evaluation horizons
    - Comprehensive logging and validation
    """
    
    def __init__(self, cutoff_date: str = "2024-06-30"):
        """
        Initialize the data splitter.
        
        Args:
            cutoff_date (str): Last date allowed in training data (YYYY-MM-DD)
        """
        self.cutoff_date = cutoff_date
        self.cutoff_datetime = pd.to_datetime(cutoff_date)
        
        logger.info(f"DataSplitter initialized with cutoff: {cutoff_date}")
    
    def split_data(self, data: pd.DataFrame, 
                   date_column: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and evaluation sets based on cutoff date.
        
        Args:
            data (pd.DataFrame): Input data with date column
            date_column (str): Name of the date column
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (training_data, evaluation_data)
        """
        try:
            # Ensure data has the date column
            if date_column not in data.columns:
                raise ValueError(f"Date column '{date_column}' not found in data")
            
            # Convert date column to datetime
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
            
            # Split data based on cutoff date
            training_mask = data[date_column] <= self.cutoff_datetime
            evaluation_mask = data[date_column] > self.cutoff_datetime
            
            training_data = data[training_mask].copy()
            evaluation_data = data[evaluation_mask].copy()
            
            # Log split statistics
            total_records = len(data)
            training_records = len(training_data)
            evaluation_records = len(evaluation_data)
            
            logger.info(f"Data split completed:")
            logger.info(f"  Total records: {total_records}")
            logger.info(f"  Training records: {training_records} ({training_records/total_records*100:.1f}%)")
            logger.info(f"  Evaluation records: {evaluation_records} ({evaluation_records/total_records*100:.1f}%)")
            
            if training_records > 0:
                logger.info(f"  Training period: {training_data[date_column].min().date()} to {training_data[date_column].max().date()}")
            if evaluation_records > 0:
                logger.info(f"  Evaluation period: {evaluation_data[date_column].min().date()} to {evaluation_data[date_column].max().date()}")
            
            return training_data, evaluation_data
            
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise
    
    def validate_split(self, training_data: pd.DataFrame, 
                      evaluation_data: pd.DataFrame,
                      date_column: str = 'date') -> bool:
        """
        Validate that there's no data leakage between training and evaluation sets.
        
        Args:
            training_data (pd.DataFrame): Training dataset
            evaluation_data (pd.DataFrame): Evaluation dataset
            date_column (str): Name of the date column
            
        Returns:
            bool: True if split is valid (no leakage), False otherwise
        """
        try:
            if len(training_data) == 0 and len(evaluation_data) == 0:
                logger.warning("Both training and evaluation datasets are empty")
                return True
            
            if len(training_data) == 0:
                logger.info("Training dataset is empty - validation passed")
                return True
                
            if len(evaluation_data) == 0:
                logger.info("Evaluation dataset is empty - validation passed")
                return True
            
            # Ensure date columns are datetime
            training_data = training_data.copy()
            evaluation_data = evaluation_data.copy()
            training_data[date_column] = pd.to_datetime(training_data[date_column])
            evaluation_data[date_column] = pd.to_datetime(evaluation_data[date_column])
            
            # Check for temporal overlap
            max_training_date = training_data[date_column].max()
            min_evaluation_date = evaluation_data[date_column].min()
            
            has_leakage = max_training_date >= min_evaluation_date
            
            if has_leakage:
                logger.error(f"❌ Data leakage detected!")
                logger.error(f"  Max training date: {max_training_date.date()}")
                logger.error(f"  Min evaluation date: {min_evaluation_date.date()}")
                logger.error(f"  Overlap: {(max_training_date - min_evaluation_date).days} days")
                return False
            
            # Check cutoff date compliance
            training_after_cutoff = training_data[training_data[date_column] > self.cutoff_datetime]
            evaluation_before_cutoff = evaluation_data[evaluation_data[date_column] <= self.cutoff_datetime]
            
            if len(training_after_cutoff) > 0:
                logger.error(f"❌ Training data contains {len(training_after_cutoff)} records after cutoff date")
                return False
            
            if len(evaluation_before_cutoff) > 0:
                logger.error(f"❌ Evaluation data contains {len(evaluation_before_cutoff)} records before cutoff date")
                return False
            
            # Validation passed
            gap_days = (min_evaluation_date - max_training_date).days
            logger.info(f"✅ Data split validation passed")
            logger.info(f"  No temporal overlap detected")
            logger.info(f"  Gap between training and evaluation: {gap_days} days")
            logger.info(f"  Training cutoff compliance: ✅")
            logger.info(f"  Evaluation cutoff compliance: ✅")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def create_time_windows(self, start_date: str, end_date: str, 
                          window_sizes: List[int] = [1, 5, 30, 90, 180, 365]) -> Dict[str, Tuple[str, str]]:
        """
        Create multiple time windows for multi-horizon evaluation.
        
        Args:
            start_date (str): Start date for window creation
            end_date (str): End date for window creation
            window_sizes (List[int]): List of window sizes in days
            
        Returns:
            Dict[str, Tuple[str, str]]: Dictionary mapping window names to (start, end) dates
        """
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            windows = {}
            
            for window_size in window_sizes:
                window_start = start_dt
                window_end = start_dt + timedelta(days=window_size)
                
                # Ensure window doesn't exceed end date
                if window_end > end_dt:
                    window_end = end_dt
                
                window_name = f"{window_size}D"
                windows[window_name] = (
                    window_start.strftime('%Y-%m-%d'),
                    window_end.strftime('%Y-%m-%d')
                )
                
                logger.debug(f"Created window {window_name}: {windows[window_name]}")
            
            logger.info(f"Created {len(windows)} time windows for multi-horizon evaluation")
            return windows
            
        except Exception as e:
            logger.error(f"Failed to create time windows: {e}")
            raise
    
    def filter_data_by_window(self, data: pd.DataFrame, 
                             window_start: str, window_end: str,
                             date_column: str = 'date') -> pd.DataFrame:
        """
        Filter data to a specific time window.
        
        Args:
            data (pd.DataFrame): Input data
            window_start (str): Window start date (YYYY-MM-DD)
            window_end (str): Window end date (YYYY-MM-DD)
            date_column (str): Name of the date column
            
        Returns:
            pd.DataFrame: Filtered data within the time window
        """
        try:
            # Convert dates
            start_dt = pd.to_datetime(window_start)
            end_dt = pd.to_datetime(window_end)
            
            # Ensure data has datetime column
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])
            
            # Filter data
            mask = (data[date_column] >= start_dt) & (data[date_column] <= end_dt)
            filtered_data = data[mask].copy()
            
            logger.debug(f"Filtered data from {window_start} to {window_end}: {len(filtered_data)} records")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Failed to filter data by window: {e}")
            raise
    
    def get_split_summary(self, training_data: pd.DataFrame, 
                         evaluation_data: pd.DataFrame,
                         date_column: str = 'date') -> Dict[str, Any]:
        """
        Get comprehensive summary of data split.
        
        Args:
            training_data (pd.DataFrame): Training dataset
            evaluation_data (pd.DataFrame): Evaluation dataset
            date_column (str): Name of the date column
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        try:
            summary = {
                "cutoff_date": self.cutoff_date,
                "total_records": len(training_data) + len(evaluation_data),
                "training": {
                    "records": len(training_data),
                    "percentage": len(training_data) / (len(training_data) + len(evaluation_data)) * 100 if (len(training_data) + len(evaluation_data)) > 0 else 0
                },
                "evaluation": {
                    "records": len(evaluation_data),
                    "percentage": len(evaluation_data) / (len(training_data) + len(evaluation_data)) * 100 if (len(training_data) + len(evaluation_data)) > 0 else 0
                }
            }
            
            # Add date ranges if data exists
            if len(training_data) > 0:
                training_data = training_data.copy()
                training_data[date_column] = pd.to_datetime(training_data[date_column])
                summary["training"]["date_range"] = {
                    "start": training_data[date_column].min().strftime('%Y-%m-%d'),
                    "end": training_data[date_column].max().strftime('%Y-%m-%d'),
                    "days": (training_data[date_column].max() - training_data[date_column].min()).days + 1
                }
            
            if len(evaluation_data) > 0:
                evaluation_data = evaluation_data.copy()
                evaluation_data[date_column] = pd.to_datetime(evaluation_data[date_column])
                summary["evaluation"]["date_range"] = {
                    "start": evaluation_data[date_column].min().strftime('%Y-%m-%d'),
                    "end": evaluation_data[date_column].max().strftime('%Y-%m-%d'),
                    "days": (evaluation_data[date_column].max() - evaluation_data[date_column].min()).days + 1
                }
            
            # Validate split
            summary["validation"] = {
                "has_leakage": not self.validate_split(training_data, evaluation_data, date_column),
                "is_valid": self.validate_split(training_data, evaluation_data, date_column)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate split summary: {e}")
            raise
    
    def save_split_metadata(self, training_data: pd.DataFrame, 
                           evaluation_data: pd.DataFrame,
                           output_dir: str = "logs/evaluation",
                           date_column: str = 'date') -> str:
        """
        Save split metadata to file for audit trail.
        
        Args:
            training_data (pd.DataFrame): Training dataset
            evaluation_data (pd.DataFrame): Evaluation dataset
            output_dir (str): Output directory
            date_column (str): Name of the date column
            
        Returns:
            str: Path to saved metadata file
        """
        try:
            import json
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate summary
            summary = self.get_split_summary(training_data, evaluation_data, date_column)
            
            # Add metadata
            metadata = {
                "split_metadata": summary,
                "generated_at": datetime.now().isoformat(),
                "splitter_version": "1.0",
                "parameters": {
                    "cutoff_date": self.cutoff_date,
                    "date_column": date_column
                }
            }
            
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metadata_file = output_path / f"data_split_metadata_{timestamp}.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Split metadata saved to: {metadata_file}")
            return str(metadata_file)
            
        except Exception as e:
            logger.error(f"Failed to save split metadata: {e}")
            raise

# Utility functions for easy access
def split_temporal_data(data: pd.DataFrame, cutoff_date: str = "2024-06-30") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick function to split data temporally.
    
    Args:
        data (pd.DataFrame): Input data with date column
        cutoff_date (str): Training/evaluation cutoff date
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (training_data, evaluation_data)
    """
    splitter = DataSplitter(cutoff_date)
    return splitter.split_data(data)

def validate_temporal_split(training_data: pd.DataFrame, evaluation_data: pd.DataFrame,
                           cutoff_date: str = "2024-06-30") -> bool:
    """
    Quick function to validate temporal split.
    
    Args:
        training_data (pd.DataFrame): Training dataset
        evaluation_data (pd.DataFrame): Evaluation dataset
        cutoff_date (str): Training/evaluation cutoff date
        
    Returns:
        bool: True if split is valid
    """
    splitter = DataSplitter(cutoff_date)
    return splitter.validate_split(training_data, evaluation_data)

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example data splitting
    try:
        # Create sample data
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(len(dates)),
            'ticker': 'SAMPLE.NS'
        })
        
        # Split data
        splitter = DataSplitter("2024-06-30")
        training, evaluation = splitter.split_data(sample_data)
        
        # Validate split
        is_valid = splitter.validate_split(training, evaluation)
        
        # Generate summary
        summary = splitter.get_split_summary(training, evaluation)
        
        print("Data splitting example completed successfully!")
        print(f"Training records: {len(training)}")
        print(f"Evaluation records: {len(evaluation)}")
        print(f"Split is valid: {is_valid}")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc() 
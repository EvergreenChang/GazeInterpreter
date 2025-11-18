import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import json
import os
import glob

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processor"""
    
    def __init__(self, data_path: str):
        """
        Initialize data processor
        
        Args:
            data_path: Data file path or directory path
        """
        self.data_path = data_path
        self.is_directory = os.path.isdir(data_path)
        self.data = self._load_data()
        
    def _load_data(self) -> Union[List[Dict], Dict[str, List[Dict]]]:
        """
        Load data
        
        Returns:
            If input is a file: List[Dict]: Data list
            If input is a directory: Dict[str, List[Dict]]: Mapping of file name to data list
        """
        try:
            if self.is_directory:
                # Process directory case
                data_dict = {}
                json_files = glob.glob(os.path.join(self.data_path, "*.json"))
                
                for json_file in json_files:
                    filename = os.path.basename(json_file)
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    data_dict[filename] = file_data
                    logger.info(f"Successfully loaded {len(file_data)} data from file {filename}")
                
                if not data_dict:
                    raise ValueError(f"No JSON file found in directory {self.data_path}")
                
                logger.info(f"Loaded {len(data_dict)} JSON files")
                return data_dict
            else:
                # Process single file case
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded {len(data)} data")
                return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_items(self, max_items: Optional[int] = None) -> Union[List[Dict], Dict[str, List[Dict]]]:
        """
        Get data items
        
        Args:
            max_items: Maximum number of data items
            
        Returns:
            If input is a file: List[Dict]: Data items list
            If input is a directory: Dict[str, List[Dict]]: Mapping of file name to data list
        """
        if self.is_directory:
            result = {}
            for filename, items in self.data.items():
                if max_items is not None:
                    result[filename] = items[:max_items]
                else:
                    result[filename] = items
            return result
        else:
            if max_items is not None:
                return self.data[:max_items]
            return self.data 
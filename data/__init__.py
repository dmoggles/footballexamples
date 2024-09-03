
"""
Module: data
This module provides functions for reading CSV files and returning their 
contents as pandas DataFrames.
Functions:
- get_data(name:str) -> pd.DataFrame: Reads a CSV file and returns its contents as a 
pandas DataFrame.
"""
import os

import pandas as pd


def get_data(name:str)->pd.DataFrame:
    """
    Reads a CSV file and returns its contents as a pandas DataFrame.
    Parameters:
    - name (str): The name of the CSV file (without the extension).
    Returns:
    - pd.DataFrame: The contents of the CSV file as a pandas DataFrame.
    """

    # get path relative to current file
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, f'{name}.csv')
    return pd.read_csv(path, index_col=0)

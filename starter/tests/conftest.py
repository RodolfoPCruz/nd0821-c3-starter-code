import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from constants import raw_file_path, cleaned_file_path
from basic_cleaning import import_data

@pytest.fixture
def param_raw_file_path():
    """
    Provides the path to load the raw dataframe

    Returns:
        str: path of the file 
    """

    return raw_file_path

@pytest.fixture
def param_raw_data():
    """
    Provides dataframe with the raw data

    Return 
        pandas dataframe: initial dataset
    """
    raw_dataframe = import_data(raw_file_path)
    return raw_dataframe

@pytest.fixture
def param_cleaned_data_path():
    """
    Provides the path of the cleaned dataframe

    Return 
        str: path of the clened dataset file
    """
    return cleaned_file_path

@pytest.fixture
def param_clened_data():
    """
    Provides the cleaned dataframe

    Return 
        pandas dataframe: cleaned dataset
    """
    cleaned_dataframe = import_data(cleaned_file_path)
    return cleaned_dataframe
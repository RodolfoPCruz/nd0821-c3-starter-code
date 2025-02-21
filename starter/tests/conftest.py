import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from constants import raw_file_path, cleaned_file_path, train_path, test_path, test_size, random_state_split, log_file_test_path
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
def param_cleaned_data():
    """
    Provides the cleaned dataframe

    Return 
        pandas dataframe: cleaned dataset
    """
    cleaned_dataframe = import_data(cleaned_file_path)
    return cleaned_dataframe

@pytest.fixture
def param_categorical_features():
    """
    Provides the list of categorical features

    Return 
        list: list of categorical features
    """
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
    return cat_features

@pytest.fixture
def param_label():
    '''
    Provide the label of the dataset

    Return
        str: column name of the expected output
    '''
    return "salary"

@pytest.fixture
def param_train_path():
    '''
    Provide the path to the train dataset

    Return
        str: path to the train dataset
    '''
    return train_path

@pytest.fixture
def param_train_data():
    '''
    Provide the train dataset

    Return
        pandas dataframe: train dataset
    '''
    train_dataframe = import_data(train_path)
    return train_dataframe

@pytest.fixture
def param_test_path():
    '''
    Provide the path to the test dataset

    Return
        str: path to the test dataset
    '''
    return test_path

@pytest.fixture
def param_test_size():
    '''
    Provide the test size

    Return
        float: proportion of the dataset that will be used for testing
    '''
    return test_size

@pytest.fixture
def param_random_state_split():
    '''
    Provide the random state for splitting the data

    Return
        int: Controls the shuffling applied to the data before applying the split. 
            Pass an int for reproducible output across multiple function calls
    '''
    return random_state_split

import logging
import pytest

@pytest.fixture(autouse=True)
def setup_logging():
    """Ensure tests log only to test.log, not the main application log."""
    logger = logging.getLogger()  # Get the root logger

    # Remove all existing handlers (prevents logging to app.log)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Disable log propagation (prevents logs from appearing in other files)
    logger.propagate = False

    # Set up test logging
    handler = logging.FileHandler(log_file_test_path, mode="w")  # Overwrite test log
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add new handler for test logs only
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Tests functions in the basic_cleaning.py file

import pytest
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from basic_cleaning import import_data
from constants import log_file_test_path


logging.basicConfig(
    filename = log_file_test_path,
    level = logging.INFO,
    force = True,
    filemode = 'w',
    format = '%(asctime)- 15s %(name)s - %(levelname)s - %(message)s')

def test_import(param_raw_file_path):
    """
    Test the import_data function

    Args:
        raw_file_path (str): path to the raw dataframe
    """
    logging.info("Testing import_data function")
    try:
        df = import_data(param_raw_file_path)
        logging.info("Sucess: Dataframe Loaded")
    except FileNotFoundError as err:
        logging.error("The file was not found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("The dataframe is empty")
        raise err
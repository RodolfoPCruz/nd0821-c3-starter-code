"""
Tests for functions in basic_cleaning.py

test_load_data - Tests if dataframe is being correctly loaded and if loaded the dataframe is not empty
test_removing_whitespaces - Tests if there are no leading or trailing spaces in the column names
test_removing_duplicates - Tests if there are no duplicated rows
test_saving_dataframe - Tests if the clenned dataframe was saved
"""

import os
import sys
import logging

from constants import log_file_test_path
from basic_cleaning import (import_data, removing_duplicates,
                            removing_whitespaces, saving_dataframe)

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'src')))


logging.basicConfig(
    filename=log_file_test_path,
    level=logging.INFO,
    force=True,
    filemode='w',
    format='%(asctime)- 15s %(name)s - %(levelname)s - %(message)s')


def test_load_data(param_raw_file_path):
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


def test_removing_whitespaces(param_raw_data):
    """
    Test removing_whitespace function

    Args:
        raw_data(pd.DataFrame): initial dataset
    """
    logging.info("Testing removing_whitespaces function")
    try:
        df = removing_whitespaces(param_raw_data)
        column_names = df.columns.to_list()
        leading_or_trailing_spaces = [name.startswith(
            " ") or name.endswith(" ") for name in column_names]
        assert not any(leading_or_trailing_spaces)
        logging.info(
            'Sucess: There are not leading or trailing spaces in the column names')
    except AssertionError as err:
        logging.error(
            "There are leading spaces or trailing spaces in the column names")
        raise err


def test_removing_duplicates(param_raw_data):
    """
    Test removing_duplicates function

    Args:
        raw_data(pd.DataFrame): initial dataset
    """
    logging.info("Testing removing_duplicates function")
    try:
        df = removing_duplicates(param_raw_data)
        assert not df.duplicated().any()
        logging.info('Sucess: There are not duplicated rows')
    except AssertionError as err:
        logging.error('There are duplicated rows')
        raise err


def test_saving_dataframe(param_raw_data, param_cleaned_data_path):
    '''
    Test saving_dataframe function

    Args:
        raw_data(pd.DataFrame): initial dataset
        cleaned

    '''
    logging.info('Testing saving_dataframe function')
    df = removing_whitespaces(param_raw_data)
    df = removing_duplicates(df)
    saving_dataframe(df, param_cleaned_data_path)

    try:
        assert os.path.exists(param_cleaned_data_path)
        logging.info('Sucess: The file was saved')
    except AssertionError as err:
        logging.error('The file was not saved')
        raise err

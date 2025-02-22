"""
Test funcion in data_splitting.py

Test if the training and testing dataframes are created and saved
"""

import os
import sys

import pandas as pd
from data_splitting import split_data
from logging_config import test_logger


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'src')))


def test_data_splitting(param_cleaned_data,
                        param_train_path,
                        param_test_path,
                        param_random_state_split,
                        param_test_size):
    '''
    Test data_splitting function

    Args:
        param_cleaned_data(pd.DataFrame): dataframe cleaned
        param_train_path(str): path to save the training dataframe
        param_test_path(str): path to save the testing dataframe
        param_random_state_split(int): random state to split the data
        param_test_size(float): proportion of the data to include in the test split

    '''
    _, _ = split_data(param_cleaned_data,
                      param_test_size,
                      param_train_path,
                      param_test_path,
                      param_random_state_split)

    train = pd.read_csv(param_train_path)
    test = pd.read_csv(param_test_path)

    test_logger.info('Testing data_splitting function')

    try:
        assert train.shape[0] > 0
        assert train.shape[1] > 0
        test_logger.info(
            'The training dataframe was saved and it is not empty')
    except AssertionError as err:
        test_logger.error("The training dataframe is empty")
        raise err
    try:
        assert test.shape[0] > 0
        assert test.shape[1] > 0
        test_logger.info('The testing dataframe was saved and it is not empty')
    except AssertionError as err:
        test_logger.error("The testing dataframe is empty")
        raise err

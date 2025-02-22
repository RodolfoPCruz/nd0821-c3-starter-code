'''
Tests for the process_data function in the data.py file. 
There are three tests:
'''

import pandas as pd
import numpy as np
from data import process_data
import os
import sys
from logging_config import test_logger


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'src')))


def test_process_data_known_categories():
    '''
    Test if the data is being correctly encoded. Some fake simple data is
    generated to test if the function produces the expected outoput.
    '''

    fake_data = pd.DataFrame({
        'name': ['Brady', 'Mannning', 'Rodgers'],
        'rings': [7, 2, 1],
        'salary': ['<15', '<15', '>20']
    })
    expected_x = np.array([[7, 1, 0, 0],
                           [2, 0, 1, 0],
                           [1, 0, 0, 1]])
    expected_y = np.array([0, 0, 1])

    x, y, _, _ = process_data(fake_data, categorical_features=[
                              'name'], label='salary')

    test_logger.info(
        'Testing process_data using fake data. All categories are known')
    try:
        assert np.array_equal(x, expected_x)
        test_logger.info('The input data is correctly encoded')
    except AssertionError as err:
        test_logger.error('The input data is not correctly encoded')
        raise err
    try:
        assert np.array_equal(y, expected_y)
        test_logger.info('The expected output is correctly encoded')
    except AssertionError as err:
        test_logger.error('The expected output is not corrected encoded')
        raise err


def test_process_data_unknown_categories():
    '''
    Test if the data is being correctly encoded. Some fake simple data is
    generated to test if the function produces the expected outoput.

    The one hot encoder is fitted and then tested with inputs containing 
    unknown categories
    '''

    fake_train_data = pd.DataFrame({
        'name': ['Brady', 'Mannning', 'Rodgers'],
        'rings': [7, 2, 1],
        'salary': ['<15', '<15', '>20']
    })

    fake_test_data = pd.DataFrame = fake_data = pd.DataFrame({
        'name': ['Mahomes', 'Mannning', 'Rodgers'],
        'rings': [3, 2, 1],
        'salary': ['>20', '<15', '>20']
    })

    expected_x = np.array([[3, 0, 0, 0],
                           [2, 0, 1, 0],
                           [1, 0, 0, 1]])
    expected_y = np.array([1, 0, 1])

    _, _, encoder, lb = process_data(
        fake_train_data, categorical_features=['name'], label='salary')
    x, y, _, _ = process_data(fake_test_data, categorical_features=['name'],
                              label='salary', training=False, encoder=encoder, lb=lb)

    test_logger.info(
        'Testing process_data using fake data. The encoder will have to deal with inknown categories')
    try:
        assert np.array_equal(x, expected_x)
        test_logger.info('The input data is correctly encoded')
    except AssertionError as err:
        test_logger.error('The input data is not correctly encoded')
        raise err
    try:
        assert np.array_equal(y, expected_y)
        test_logger.info('The expected output is correctly encoded')
    except AssertionError as err:
        test_logger.error('The expected output is not corrected encoded')
        raise err


def test_process_data_using_real_data(
        param_train_data,
        param_categorical_features,
        param_label):
    '''
    Test the process_data function using real data. The output must not
    have categorical features
    '''


    x, y, _, _ = process_data(
        param_train_data, categorical_features=param_categorical_features, label=param_label)
    test_logger.info('Testing process_data using real data')

    try:
        assert np.issubdtype(x.dtype, np.number)
        test_logger.info(
            'The input data is correctly encoded. There are only numerical values')
    except AssertionError as err:
        test_logger.error(
            'The input data is not correctly encoded. There are non-numerical values')
        raise err

    try:
        assert np.issubdtype(y.dtype, np.number)
        test_logger.info(
            'The expected output is correctly encoded. There are only numerical values')
    except AssertionError as err:
        test_logger.error(
            'The expected output is not correctly encoded. There are non-numerical values')
        raise err

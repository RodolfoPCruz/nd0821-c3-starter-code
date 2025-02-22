"""
Tests from the functions in model.py
"""


import os
import sys
from model import train_model, inference, compute_model_metrics
import numpy as np
from logging_config import test_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'src')))


def test_train_model_valid_input(param_number_features):
    """
    Test for the train_model
    Test if a random forest classifier is trained

    """

    number_of_samples = 10
    x = np.random.rand(number_of_samples, param_number_features)
    y = np.random.randint(2, size=number_of_samples)

    model = train_model(x, y)

    test_logger.info('Testing train_model with valid inputs')

    try:
        assert isinstance(model, RandomForestClassifier)
        test_logger.info(
            'Sucess: A RandomForestClassifier instance was created')
    except AssertionError as err:
        test_logger.error('A RandomForestClassifier instance was not created')
        raise err

    try:
        assert hasattr(model, 'feature_importances_')
        test_logger.info(
            'Sucess: A RandomForestClassifier was trained '
            'and it has the feature_importances_ attribute')

    except AssertionError as err:
        test_logger.error(
            'The model was not trained. '
            'It does not have the feature_importances_ attribute')
        raise err


def test_train_model_reproducibility(param_number_features):
    """
    Test for train model function
    Test if the same model is generated after
        training it two times using the same training data
    """

    number_of_samples = 10
    x = np.random.rand(number_of_samples, param_number_features)
    y = np.random.randint(2, size=number_of_samples)

    test_logger.info('Testing train_model for reproducibility')

    model_1 = train_model(x, y)
    model_2 = train_model(x, y)

    try:
        assert np.array_equal(
            model_1.feature_importances_,
            model_2.feature_importances_)
        test_logger.info('Sucess: The two models are equal')

    except AssertionError as err:
        test_logger.error("The two models are not equal")
        raise err


def test_inference_valid_inputs(param_number_features):
    """
    Test if the inference function is generating the expected
    number of predictions and
    wether the predictions are in the correct format
    """

    number_of_samples = 10
    x = np.random.rand(number_of_samples, param_number_features)
    y = np.random.randint(2, size=number_of_samples)

    test_logger.info('Testing inference with valid inputs')

    model = train_model(x, y)
    y_pred = inference(model, x)

    try:
        assert isinstance(y_pred, np.ndarray)
        test_logger.info('Sucess: A numpy array was returned')
    except AssertionError as err:
        test_logger.error('The output of the function is not a numpy array')
        raise err

    try:
        assert len(y_pred) == number_of_samples
        test_logger.info(
            'Success: The correct number of predictions were generated')
    except AssertionError as err:
        test_logger.error(
            "The number of predctions is different of "
            "the number of input samples")
        raise err


def test_compute_model_metrics_valid_inputs():
    '''
    Test wether the metrics calculated by the
    compute_model_metrics function are correct
    '''

    y = [0, 0, 1, 0]
    preds = [0, 0, 1, 0]

    test_logger.info('Testing compute_model_metrics with valid inputs')

    precision_1 = precision_score(y, preds)
    recall_1 = recall_score(y, preds)
    fbeta_1 = fbeta_score(y, preds, beta=1)

    precision_2, recall_2, fbeta_2 = compute_model_metrics(y, preds)

    try:
        assert precision_1 == precision_2
        test_logger.info('The precision is correct')
    except AssertionError as err:
        test_logger.error('The precision is not correct')
        raise err

    try:
        assert recall_1 == recall_2
        test_logger.info('The recall is correct')
    except AssertionError as err:
        test_logger.error('The recall is not correct')
        raise err

    try:
        assert fbeta_1 == fbeta_2
        test_logger.info('The fbeta is correct')
    except AssertionError as err:
        test_logger.error('The fbeta is not correct')
        raise err


def test_compute_model_metrics_perfect_classsification():
    """
    Test wether the metrics calculated by the
    compute_model_metrics function are correct
    when all the predctions made by the model are true
    """

    y = [1, 0, 0, 1]
    preds = [1, 0, 0, 1]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    try:
        assert precision == 1
        test_logger.info(
            'Sucess: Precision is equal to 1 when all '
            'samples are correctly classified')
    except AssertionError as err:
        test_logger.error(
            'Precision is not correct. Precision ssould be equal '
            'to 1 when all samples are correctly classified')
        raise err

    try:
        assert recall == 1
        test_logger.info(
            'Sucess: Recall is equal to 1 when all '
            'samples are correctly classified')
    except AssertionError as err:
        test_logger.error(
            'Recall is not correct. Recall should be equal to 1 '
            'when all samples are correctly classified')
        raise err

    try:
        assert fbeta == 1
        test_logger.info(
            'Sucess: Fbeta is equal to 1 when '
            'all samples are correctly classified')
    except AssertionError as err:
        test_logger.error(
            'Fbeta is not correct. Fbeta should be equal '
            'to 1 when all samples are correctly classified')
        raise err


def test_compute_model_metrics_division_by_zero():
    """
    Test wether the metrics calculated by the
    compute_model_metrics function are correct
    when there not sample of the positive class
    """

    y = [0, 0, 0, 0]
    preds = [0, 0, 0, 0]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    try:
        assert precision == 1
        test_logger.info(
            'Sucess: Precision is equal to 1 when '
            'there are no samples from the positive class')
    except AssertionError as err:
        test_logger.error(
            'Precision is not correct. Precision ssould be equal '
            'to 1 there are no samples from the positive class')
        raise err

    try:
        assert recall == 1
        test_logger.info(
            'Sucess: Recall is equal to 1 when there are '
            'no samples from the positive class')
    except AssertionError as err:
        test_logger.error(
            'Recall is not correct. Recall should be equal to 1 '
            'there are no samples from the positive class')
        raise err

    try:
        assert fbeta == 1
        test_logger.info(
            'Sucess: Fbeta is equal to 1 when there are '
            'no samples from the positive class')
    except AssertionError as err:
        test_logger.error(
            'Fbeta is not correct. Fbeta should be equal to 1 '
            'there are no samples from the positive class')
        raise err

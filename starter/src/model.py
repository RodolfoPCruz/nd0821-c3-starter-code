"""
Script to train a machine learning models, generate predictions and calculate metrics.

"""


from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from constants import random_state_model, model_name, classification_report_path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must not be None")

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("X_train and y_train must not be empty")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must have the same number of samples ")

    random_forest_model = RandomForestClassifier(
        random_state=random_state_model)
    random_forest_model.fit(X_train, y_train)

    return random_forest_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    if len(y) == 0 or len(preds) == 0:
        raise ValueError('The inputs can not be empty')

    if len(y) != len(preds):
        raise ValueError('y and preds must have the same number of samples')

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : random forest model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    if model is None:
        raise ValueError('Model can not be None')

    if X is None:
        raise ValueError('Input data can not be None')

    if len(X) == 0:
        raise ValueError('Input data can not be empty')

    if not hasattr(model, 'predict'):
        raise ValueError('The model does not have the predict attribute')

    preds = model.predict(X)
    return preds

def classification_report_image(y_train, y_train_preds,
                                y_test, y_test_preds):
    """
    Produces classification report for training and testing results and stores report as image
    in images folder

    Args:
            y_train       - expected outputs for training data
            y_train_preds - predictions for training data
            y_test        - expected outputs for test data
            y_test_preds  - predictions for test data

    Returns:
             None
    """

    
    plt.rc('figure', figsize=(8, 8))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
    # approach
    plt.text(0.01, 1, str(model_name + ' Train'),
                {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.2, str(
            classification_report(
                y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.5, str(model_name + ' Test'),
                {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(classification_report_path)
    plt.clf()

'''
Script to train machine learning model.
'''

import os
import sys
from pickle import dump

import numpy as np
import pandas as pd

from constants import cat_features, label, model_path, test_path, train_path
from data import process_data
from model import (compute_model_metrics, create_save_confusion_matrix,
                   inference, save_classification_report_image, train_model)

sys.path.append(os.path.abspath("ml"))


# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Process the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(train,
                                             categorical_features=cat_features,
                                             label=label,
                                             training=True
                                             )
print(type(X_train))

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features,
                                    label=label,
                                    training=False,
                                    encoder=encoder,
                                    lb=lb)

# Train model.
random_forest_model = train_model(X_train, y_train)
# Save model
with open(model_path, "wb") as f:
    dump(random_forest_model, f, protocol=5)
# Inference
y_pred_train = inference(random_forest_model, X_train)
y_pred_test = inference(random_forest_model, X_test)
# Metrics
precision_train, recall_train, fbeta_train = compute_model_metrics(
    y_train, y_pred_train)
#Classification report
save_classification_report_image(y_train, y_pred_train,
                                 y_test, y_pred_test)
#confusion matrix
create_save_confusion_matrix(y_test, y_pred_test)

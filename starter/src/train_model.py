'''
Script to train machine learning model.
'''

from sklearn.model_selection import train_test_split

import sys
import os
from pickle import dump
from constants import model_path

sys.path.append(os.path.abspath("ml"))

# Add the necessary imports for the starter code.
from data import process_data
import pandas as pd
from constants import train_path, test_path, cat_features, label
from model import train_model, compute_model_metrics, inference

#Load data
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# Process the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(train, 
                                             categorical_features = cat_features, 
                                             label = label, 
                                             training = True
)



# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features = cat_features, 
                                             label = label, 
                                             training = False,
                                             encoder = encoder,
                                             lb = lb)

# Train model.
random_forest_model = train_model(X_train, y_train)
#Save model
with open(model_path, "wb") as f:
    dump(random_forest_model, f, protocol=5)
#Inference
y_pred_train = inference(random_forest_model, X_train)
#Metrics
precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, y_pred_train)
print(precision_train)
print(recall_train)
print(fbeta_train)
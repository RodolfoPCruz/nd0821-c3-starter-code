"""
constants.py
----------------
----------------
Constants that will be used 
Author: Rodolfo Cruz
Date: 2025-02-18
"""

# path to the raw dataframe
raw_file_path = "../data/census.csv"

#path to the clenaed datframe
cleaned_file_path = "../data/cleaned_census.csv"

#path to the log file
log_file_path = '../logs/logs.log'

#path to the log file created when testing
log_file_test_path = '../logs/logs_tests.log'

# Random state for splitting the data
random_state_split = 42

#Test size
test_size = 0.2

# path to the train dataframe
train_path = "../data/train.csv"

# path to the test dataframe
test_path = "../data/test.csv"

#path to the model
model_path = "../model/random_forest_model.pkl"

#categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]

#column name of the expected output
label = "salary"

# Random state for ml model
random_state_model = 42

# Model Name
model_name = 'Random Forest'

# Path classification report
classification_report_path = '../images/classification_report.png'
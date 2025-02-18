# Script to clean the dataset.

import os
import pandas as pd


#Load dataset
data_path = os.path.abspath(os.path.join(os.getcwd(), "..", "data/census.csv"))
data = pd.read_csv(data_path)

#Removing whitespaces in the column names
column_names = data.columns.to_list()
column_names = [name.strip() for name in column_names]
data.columns = column_names

#Removing duplicates
data = data.drop_duplicates(data)

#Saving dataframe
data.to_csv(path_or_buf = os.path.abspath(os.path.join(os.getcwd(), "..", "data/cleaned_census.csv")),
            index = False )
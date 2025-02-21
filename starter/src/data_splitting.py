"""
Script to split the dataset into training and testing sets.
"""
from logging_config import app_logger 
import pandas as pd
from sklearn.model_selection import train_test_split

from constants import (cleaned_file_path, log_file_path, random_state_split,
                       test_path, test_size, train_path)


def load_cleaned_data(pth: str) -> pd.DataFrame:
    """
    Load cleaned data from the path

    Args:
        pth (str): path to the cleaned data

    Returns:
        df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def split_data(
        df: pd.DataFrame,
        test_proportion: float,
        train_pth: str,
        test_pth: str,
        random_state: float) -> pd.DataFrame:
    """
    Split the data into training and testing sets

    Args:
        df (pd.DataFrame): the dataframe to split
        test_size (float): the proportion of the data to include in the test split
        pth (str): the path to save the split data

    Returns:
        train_df: the training set
        test_df: the testing
    """
    train_df, test_df = train_test_split(
        df, test_size = test_proportion, random_state=random_state)
    app_logger.info('Training and testing dataframes created')
    train_df.to_csv(train_pth, index=False)
    test_df.to_csv(test_pth, index=False)
    app_logger.info('Training and testing dataframes saved')
    return train_df, test_df


if __name__ == "__main__":
    data = load_cleaned_data(cleaned_file_path)
    x_train, x_test = split_data(
        data, test_size, train_path, test_path, random_state_split)

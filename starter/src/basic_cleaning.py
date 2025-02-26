"""
Script to clean the dataset.
 - Removal o duplicate rows;
 - Removal o leading and trailing spaces in the column names.
"""
import pandas as pd
from logging_config import app_logger

from constants import cleaned_file_path, raw_file_path

app_logger.info("Logs for basic_cleaning.py")


def import_data(pth: str) -> pd.DataFrame:
    """
    Returns dataframe for the csv found at path

    Args:
        pth (str): a path to the csv

    Returns
        df: pandas dataframe
    """
    df = pd.read_csv(pth)
    app_logger.info("File loaded")
    return df


def removing_whitespaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove whitespaces in the column names

    Args:
        df: a pandas dataframe

    Returns:
        df: a pandas dataframe without whitespace in is column names
    """
    column_names = df.columns.to_list()
    column_names = [name.strip() for name in column_names]
    df.columns = column_names
    app_logger.info("Whitespaces removed from column names")
    return df

# Removing duplicates


def removing_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated rows

    Args:
        df: a pandas dataframe

    Returns:
        df: a pandas dataframe without duplicated rows
    """
    initial_row = df.shape[0]
    df = df.drop_duplicates()
    removed_rows = initial_row - df.shape[0]
    app_logger.info(f"{removed_rows} duplicated rows removed")

    return df


def saving_dataframe(df: pd.DataFrame, pth: str):
    '''
    Save dataframe to a csv file

    Args:
        df: a pandas dataframe
        pth: a path to save the dataframe
    '''
    df.to_csv(path_or_buf=pth, index=False)
    app_logger.info("Cleaned dataframe saved")


if __name__ == "__main__":
    data = import_data(raw_file_path)
    data = removing_whitespaces(data)
    data = removing_duplicates(data)
    saving_dataframe(data, cleaned_file_path)

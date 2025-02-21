"""
Script to clean the dataset.
 - Removal o duplicate rows;
 - Removal o leading and trailing spaces in the column names.
"""
import logging

import pandas as pd

from constants import cleaned_file_path, log_file_path, raw_file_path

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)- 15s %(name)s - %(levelname)s - %(message)s')


def import_data(pth: str) -> pd.DataFrame:
    """
    Returns dataframe for the csv found at path

    Args:
        pth (str): a path to the csv

    Returns
        df: pandas dataframe
    """
    df = pd.read_csv(pth)
    logging.info("File loaded")
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
    logging.info("Whitespaces removed from column names")
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
    df = df.drop_duplicates()
    logging.info("Duplicated rows removed from the dataframe")

    return df


def saving_dataframe(df: pd.DataFrame, pth: str):
    '''
    Save dataframe to a csv file

    Args:
        df: a pandas dataframe
        pth: a path to save the dataframe
    '''
    df.to_csv(path_or_buf=pth, index=False)
    logging.info("Cleaned dataframe saved")


if __name__ == "__main__":
    data = import_data(raw_file_path)
    data = removing_whitespaces(data)
    data = removing_duplicates(data)
    saving_dataframe(data, cleaned_file_path)

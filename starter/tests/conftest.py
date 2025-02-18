import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from constants import raw_file_path


@pytest.fixture

def param_raw_file_path():
    """
    Provides the path to load the raw dataframe

    Returns:
        str: path of the file 
    """

    return raw_file_path
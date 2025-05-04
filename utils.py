import os
import glob
import pandas as pd

def is_positive_integer_col(series: pd.Series) -> bool:
    '''Helper function that identifies if the feature contains only positive integers (categorical labels)'''
    # Drop missing values
    clean_series = series.dropna()
    # Check that each value is positive and has no decimal part
    return ((clean_series > 0) & (clean_series % 1 == 0)).all()


def clear_files(extensions):
    '''Delete all files with specified extensions in the current directory'''
    for ext in extensions:
        files = glob.glob(f'*.{ext}')
        for file in files:
            os.remove(file)
            print(f"Deleted file: {file}")


def ext_exists(directory, ext):
    '''Checks if a file of particular extension exists in the directory'''
    for filename in os.listdir(directory):
        if filename.endswith(f".{ext}"):
            return True
    return False
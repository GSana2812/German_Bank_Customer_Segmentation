import pandas as pd
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler


def drop_first_column(data: pd.DataFrame, col_name: str)-> pd.DataFrame:

    """
        Remove a specified column from a DataFrame and return the modified DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to be removed.

        Returns:
        pd.DataFrame: The DataFrame with the specified column removed.
        """

    return data.drop(columns = col_name, axis=1)

def rename(data: pd.DataFrame, col_name: str)-> pd.DataFrame:

    """
        Rename a specified column in a DataFrame to 'Credit_amount' and return the modified DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to be renamed.

        Returns:
        pd.DataFrame: The DataFrame with the specified column renamed to 'Credit_amount'.
        """

    return data.rename(columns = {col_name: 'Credit_amount'})

def fill_nan_values(data: pd.DataFrame, col_name: str)->None:

    """
        Fill missing values in a specified column of a DataFrame based on data type.

        Parameters:
        data (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to fill.

        Note:
        - If the column data type is 'object', it fills missing values with the most common value.
        - If the column data type is 'int', it fills missing values with the mean.

        The function modifies the DataFrame in place and does not return a new DataFrame.
        """

    if col_name in data.columns[data.dtypes == 'object']:
        data[col_name] = data[col_name].fillna(data[col_name].value_counts().sort_values(ascending=False)[0])
    if col_name in data.columns[data.dtypes == 'int']:
        data[col_name] = data[col_name].fillna(data[col_name].mean())

def remove_skewness(data: pd.DataFrame, cols: List[str])-> pd.DataFrame:

    """
        Apply a logarithmic transformation to specified columns to reduce skewness and return the modified DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame.
        cols (List[str]): A list of column names to apply the transformation to.

        Returns:
        pd.DataFrame: The DataFrame with specified columns transformed using a logarithmic function.
        """

    data[cols] = np.log(data[cols])
    return data

def scale_features(data: pd.DataFrame)-> pd.DataFrame:
    """
       Standardize (scale) the features in a DataFrame using StandardScaler and return the modified DataFrame.

       Parameters:
       data (pd.DataFrame): The input DataFrame.

       Returns:
       pd.DataFrame: The DataFrame with features scaled using StandardScaler.
       """

    scaler = StandardScaler()
    df = scaler.fit_transform(data)
    return pd.DataFrame(data= df, columns = data.columns)




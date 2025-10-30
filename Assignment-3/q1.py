# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from ucimlrepo import fetch_ucirepo
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def downloading_data():
    ''' Downloading data'''
    # fetch dataset
    adult_income = fetch_ucirepo(id=2)
    # data (as pandas dataframes)
    X = adult_income.data.features # shape = (48842, 14)
    y = adult_income.data.targets # shape = (48842, 1)

    # Replace '?' values with NaN
    X = X.replace('?', np.nan)

    # Make labels uniform
    y = y.replace('<=50K.', '<=50K')
    y = y.replace('>50K.', '>50K')

    # metadata
    print(adult_income.metadata)
    # variable information
    print(adult_income.variables)

    # Remove the 'fnlwgt' column
    X = X.drop(columns=['fnlwgt', 'education'])

    return X, y


def data_exploration(X, y):
    """
    Using the data provided calculate: 
    - avg_age --> average age of the individuals, 
    - women_percent --> Percentage of women in the dataset (0-100 %),
    - income_percent --> Percentage of individuals earning more than $50K (0-100 %),
    - missing_values_percent --> Percentage of missing values in the dataset (0-100 %),

    Output: (n_records, n_subscriber, subscriber_percent) -> Tuple of integers
    """
    # TODO : write your code here to calculate the averages and percentages
    avg_age = None
    women_percent = None
    income_percent = None
    missing_values_percent = None
    
    # TODO: plot a scatter plot with features "hours-per-week" and "age" on the two axes, with samples labeled according to this task (i.e.,  $>\$50K$ or $\leq\$50K$)
    # Hint: Use plt.scatter() with appropriate parameters

    return avg_age, women_percent, income_percent, missing_values_percent

def data_imputation(X):
    """
    Impute the missing values in the dataset.
    Input: X: features (pd.DataFrame) with shape = (48842, 14)
    Output: X: features_imputed (pd.DataFrame) with shape = (48842, 14)
    """
    # TODO : write imputation here
    # Hint: Use SimpleImputer with strategy='most_frequent'
    X_imputed = None
    return X_imputed

def feature_encoding(X):
    """
    One-hot encode the 'features'.
    Input: X: features (pd.DataFrame)
    Output: X: features_encoded (pd.DataFrame)
    """
    # TODO : write encoding here
    # Hint: Identify categorical columns and use OneHotEncoder
    # Keep numerical columns and concatenate with encoded categorical columns
    X_final = None
    return X_final


def encode_label(y):
    """
    Encode the 'labels' data to numerical values.
    Input: y: labels (pd.DataFrame) with shape = (48842, 1)
    Output: y: labels_int (pd.DataFrame) with shape = (48842, 1)
    """
    # TODO : write encoding here
    # Hint: Use LabelEncoder to convert '<=50K' to 0 and '>50K' to 1
    y_encoded = None
    return y_encoded



def data_preprocessing():
    # First download data
    X, y = downloading_data()
    # convert categorical to numerical
    X = data_imputation(X)
    X = feature_encoding(X)
    y = encode_label(y)
    return X, y


if __name__ == "__main__":
    
    X, y = data_preprocessing()

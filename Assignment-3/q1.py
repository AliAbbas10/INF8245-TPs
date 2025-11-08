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
    avg_age = X['age'].mean()
    women_percent = (X['sex'] == 'Female').mean() * 100
    y_values = y.iloc[:, 0] if hasattr(y, 'iloc') else y
    income_percent = (y_values == '>50K').mean() * 100
    missing_values_percent = X.isnull().mean().mean() * 100
    
    # TODO: plot a scatter plot with features "hours-per-week" and "age" on the two axes, with samples labeled according to this task (i.e.,  $>\$50K$ or $\leq\$50K$)
    # Hint: Use plt.scatter() with appropriate parameters
    plt.figure(figsize=(10,5))
    y_values = y.iloc[:, 0] if hasattr(y, 'iloc') else y
    
    low_fifty = y_values == '<=50K'
    high_fifty = y_values == '>50K'
    plt.scatter(X.loc[low_fifty, 'age'], X.loc[low_fifty, 'hours-per-week'], c='blue', s=10, alpha=0.6, label='Income ≤ $50K')
    plt.scatter(X.loc[high_fifty, 'age'], X.loc[high_fifty, 'hours-per-week'], c='red', s=10, alpha=0.6, label='Income > $50K')

    plt.xlabel('Age')
    plt.ylabel('Hours per Week')
    plt.title('Hours per Week vs Age by Income')
    plt.legend()
    plt.savefig('q1_scatter_plot.png')
    plt.close()

    return avg_age, women_percent, income_percent, missing_values_percent

def data_imputation(X):
    """
    Impute the missing values in the dataset.
    Input: X: features (pd.DataFrame) with shape = (48842, 14)
    Output: X: features_imputed (pd.DataFrame) with shape = (48842, 14)
    """
    # actually 12 features
    # TODO : write imputation here
    # Hint: Use SimpleImputer with strategy='most_frequent'
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
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
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_categorical_encoded = pd.DataFrame(one_hot_encoder.fit_transform(X[categorical_cols]), columns=one_hot_encoder.get_feature_names_out(categorical_cols),index=X.index)
    X_final = pd.concat([X[numerical_cols], X_categorical_encoded], axis=1)
    return X_final


def encode_label(y):
    """
    Encode the 'labels' data to numerical values.
    Input: y: labels (pd.DataFrame) with shape = (48842, 1)
    Output: y: labels_int (pd.DataFrame) with shape = (48842, 1)
    """
    # TODO : write encoding here
    # Hint: Use LabelEncoder to convert '<=50K' to 0 and '>50K' to 1
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y.values.ravel()),  name=y.columns[0], index=y.index)
    return y_encoded



def data_preprocessing():
    # First download data
    X, y = downloading_data()
    # exploring dfata missing ?
    print("Results of data exploration")
    avg_age, women_percentage, income_percentage, missing_values_percentage = data_exploration(X, y)
    print(f"\nAverage Age: {avg_age:.2f} years")
    print(f"Percentage of Women: {women_percentage:.2f}%")
    print(f"Percentage earning > $50K: {income_percentage:.2f}%")
    print(f"Percentage of Missing Values: {missing_values_percentage:.2f}%")
    
    # convert categorical to numerical
    X = data_imputation(X)
    X = feature_encoding(X)
    y = encode_label(y)
    return X, y


if __name__ == "__main__":
    
    X, y = data_preprocessing()
    print(f"{X.shape} features")
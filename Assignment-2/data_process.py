import pandas as pd
import numpy as np
import utils


def preprocess_mnist_data(train_file_path: str, test_file_path: str):
    """Preprocess the MNIST data from CSV files.
    Args:
        train_file_path (str): Path to the training CSV file.
        test_file_path (str): Path to the testing CSV file.

    Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]: Preprocessed training and testing data, and their statistics.

    Expected types/shapes:
        - X_train: (60000, 784), dtype=np.float16, to save on memory
        - y_train: (60000,)
        - X_test: (10000, 784), dtype=np.float16
        - y_test: (10000,)
        - mean: float
        - std: float
    """
    X_train, y_train, X_test, y_test = None, None, None, None
    mean, std = None, None

    # TODO: Implement data preprocessing steps

    data_test = pd.read_csv(test_file_path, header=None).values
    data_train = pd.read_csv(train_file_path, header=None).values
    
    y_test= data_test[:, 0]
    y_train = data_train[:, 0]

    X_test = data_test[:, 1:]
    X_train = data_train[:, 1:]

    mean = np.mean(X_train)
    std = np.std(X_train)

    # Scaling
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Expect ooutput is np.float16 
    X_train = X_train.astype(np.float16)
    X_test = X_test.astype(np.float16)

    # tetsing with visualize image
    # utils.visualize_image(X_train[0], 'train')


    return X_train, y_train, X_test, y_test, mean, std


def preprocess_credit_card(train_file_path: str, test_file_path: str):
    """
        This function should be very similar to the preprocess mnist data function except that we
        have a header, the classes column has name Class and you can discard the column:
    • Open the pair of csv files as input using pandas, keeping the header and using
    column id as our index
    • extract the column Class for both,
    • perform scaling column wise: substract, for each column, its mean and divide
    by its standard deviation, as computed on the train set.
    • return the result as numpy arrays
        Args:
            train_file_path (str): Path to the training CSV file.
            test_file_path (str): Path to the testing CSV file.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]: Preprocessed training and testing data, and their statistics.
        Expected types/shapes:
            - X_train: (n_samples_train, n_features), dtype=np.float32
            - y_train: (n_samples_train,)
            - X_test: (n_samples_test, n_features), dtype=np.float32
            - y_test: (n_samples_test,)
            - mean: (n_features,), dtype=np.float32
            - std: (n_features,), dtype=np.float32
    """

    X_train, y_train, X_test, y_test = None, None, None, None
    mean, std = None, None

    data_test = pd.read_csv(test_file_path, index_col='id')
    data_train = pd.read_csv(train_file_path, index_col='id')

    y_test = data_test['Class'].values
    y_train = data_train['Class'].values

    X_test = data_test.drop('Class', axis=1).values
    X_train = data_train.drop('Class', axis=1).values

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Scaling column wise
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Expect ooutput is np.float32 
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return X_train, y_train, X_test, y_test, mean, std

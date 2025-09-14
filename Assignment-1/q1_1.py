import numpy as np


# Part (a)
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias column of ones as the first column of X."""
    # WRITE YOUR CODE HERE...

# Part (b)
def linear_regression_optimize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS solution"""
    # WRITE YOUR CODE HERE...

# Part (c)
def ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lamb: float) -> np.ndarray:
    """Closed-form Ridge regression."""
    # WRITE YOUR CODE HERE...

# Part (e)
def weighted_ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lambda_vec: np.ndarray) -> np.ndarray:
    """Weighted Ridge regression solution."""
    # WRITE YOUR CODE HERE...

# Part (f)
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute predictions: y_hat = X w"""
    # WRITE YOUR CODE HERE...

# Part (f)
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error"""
    # WRITE YOUR CODE HERE...



# Remove the following line if you are not using it:
if __name__ == "__main__":

    # If you want to test your functions, write your code here.
    # If you write it outside this snippet, the autograder will fail!

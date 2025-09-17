import numpy as np


# Part (a)
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias column of ones as the first column of X."""
    # WRITE YOUR CODE HERE...
    
    # new_X = []
    # for row in X: 
    #     new_row = np.insert(row, 0 , 1)
    #     new_X.append(new_row)

    ones_columns = np.ones((X.shape[0], 1))
    X = np.concatenate((ones_columns, X), axis=1)

    return X

# Part (b)
def linear_regression_optimize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS solution"""
    # WRITE YOUR CODE HERE...
    X_T = X.T
    w = np.dot(X_T, X)
    w = np.linalg.inv(w)
    w = np.dot(w, X_T)
    w_star = np.dot(w, y)
    return w_star.flatten()


# Part (c)
def ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lamb: float) -> np.ndarray:
    """Closed-form Ridge regression."""
    # WRITE YOUR CODE HERE...
    X_T = X.T
    I = np.eye(X.shape[1])
    w = np.dot(X_T, X) + np.dot(lamb, I)
    w = np.linalg.inv(w)
    w = np.dot(w, X_T)
    w_star = np.dot(w, y)
    return w_star.flatten()

# Part (e)
def weighted_ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lambda_vec: np.ndarray) -> np.ndarray:
    """Weighted Ridge regression solution."""
    # WRITE YOUR CODE HERE...
    X_T = X.T
    w = np.dot(X_T, X) + np.diag(lambda_vec)
    w = np.linalg.inv(w)
    w = np.dot(w, X_T)
    w_star = np.dot(w, y)
    
    return w_star.flatten()

# Part (f)
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute predictions: y_hat = X w"""
    # WRITE YOUR CODE HERE...
    return np.dot(X, w)

# Part (f)
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error"""
    # WRITE YOUR CODE HERE...

    # testing error gradescope
    # y = y.flatten()
    # y_hat = y_hat.flatten()
    
    n = len(y)
    sum= 0
    for i in range(n):
        sum += (y[i] - y_hat[i]) ** 2
    RMSE = (sum / n) ** 0.5
    return float(RMSE)



# # Remove the following line if you are not using it:
# if __name__ == "__main__":

#     # If you want to test your functions, write your code here.
#     # If you write it outside this snippet, the autograder will fail!
#     pass
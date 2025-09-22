import numpy as np
from q1_1 import rmse, ridge_regression_optimize, data_matrix_bias


# Part (a)
def cv_splitter(X, y, k):
    """
    Splits data into k folds for cross-validation.
    Returns a list of tuples: (X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    """
    # WRITE YOUR CODE HERE...

    data_size = X.shape[0]
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    fold_split_size = data_size // k
    folds = []
    for i in range(k):
        test_start_index = i * fold_split_size
        test_end_index = (i + 1) * fold_split_size if i != k - 1 else data_size

        X_val_fold = X_shuffled[test_start_index:test_end_index]
        y_val_fold = y_shuffled[test_start_index:test_end_index]

        X_train_fold = np.concatenate((X_shuffled[:test_start_index], X_shuffled[test_end_index:]), axis=0)
        y_train_fold = np.concatenate((y_shuffled[:test_start_index], y_shuffled[test_end_index:]), axis=0)

        folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        
    return folds




# Part (b)
def MAE(y, y_hat):
    # WRITE YOUR CODE HERE...
    # can i use np.mean??????
    return np.mean(np.abs(y - y_hat))




def MaxError(y, y_hat):
    # WRITE YOUR CODE HERE...
    # can i use np.max??????
    return np.max(np.abs(y - y_hat))




# Part (c)
def cross_validate_ridge(X, y, lambda_list, k, metric):
    """
    Performs k-fold CV over lambda_list using the given metric.
    metric: one of "MAE", "MaxError", "RMSE"
    Returns the lambda with best average score and a dictionary of mean scores.
    """
    # WRITE YOUR CODE HERE...
    folds = cv_splitter(X, y, k)
    #  metric operation switch case 
    if metric == "MAE":
        metric_op = MAE
    elif metric == "MaxError":
        metric_op = MaxError
    elif metric == "RMSE":
        metric_op = rmse

    mean_dict = {}

    for lamb in lambda_list:
        scores = []
        for X_train, y_train, X_test, y_test in folds:
            w_ridge = ridge_regression_optimize(X_train, y_train, lamb)
            y_pred = np.dot(X_test, w_ridge)
            scores.append(metric_op(y_test, y_pred))
        mean_dict[lamb] = np.mean(scores)

    best_lambda = min(mean_dict, key = mean_dict.get)
    return best_lambda, mean_dict



# # Remove the following line if you are not using it:
# if __name__ == "__main__":
#     # If you want to test your functions, write your code here.
#     # If you write it outside this snippet, the autograder will fail!

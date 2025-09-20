from q2_1 import *
import pandas as pd
import numpy as np

# Write your code here ...
# Not autograded — function names and structure are flexible.

X_train_arr = pd.read_csv('X_train.csv').values
X_test_arr = pd.read_csv('X_test.csv').values
Y_train = pd.read_csv('y_train.csv').values
Y_test = pd.read_csv('y_test.csv').values

lambda_vec = np.array([0.01, 0.1, 1, 1, 10, 100])

X_train = data_matrix_bias(X_train_arr)
X_test = data_matrix_bias(X_test_arr)

best_lambda_MAE, mean_scores_MAE = cross_validate_ridge(X_train, Y_train, lambda_vec, 5, "MAE")
best_lambda_MaxError, mean_scores_MaxError = cross_validate_ridge(X_train, Y_train, lambda_vec, 5, "MaxError")
best_lambda_RMSE, mean_scores_RMSE = cross_validate_ridge(X_train, Y_train, lambda_vec, 5, "RMSE")


results = pd.DataFrame({
    'Lambda vector': lambda_vec,
    'Mean MAE': [mean_scores_MAE[lam] for lam in lambda_vec],
    'Mean MaxError': [mean_scores_MaxError[lam] for lam in lambda_vec],
    'Mean RMSE': [mean_scores_RMSE[lam] for lam in lambda_vec]
})

results.to_csv("results/cross_validation_results.csv")

print("\nCross-Validation Results:")
print(results)
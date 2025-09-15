import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q1_1 import (
    data_matrix_bias,
    linear_regression_optimize,
    ridge_regression_optimize,
    weighted_ridge_regression_optimize,
    predict,
    rmse
)

# Write your code here ...
# Not autograded — function names and structure are flexible.

X_train_arr = pd.read_csv('X_train.csv').values
X_test_arr = pd.read_csv('X_test.csv').values
Y_train = pd.read_csv('y_train.csv').values
Y_test = pd.read_csv('y_test.csv').values

X_train = data_matrix_bias(X_train_arr)
X_test = data_matrix_bias(X_test_arr)

w_ord_least_squares = linear_regression_optimize(X_train, Y_train)
w_ridge = ridge_regression_optimize(X_train, Y_train, 1.0)
lambda_vec = np.array([0.01, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
w_weighted_ridge = weighted_ridge_regression_optimize(X_train, Y_train, lambda_vec)

y_pred_ord_least_squares = predict(X_test, w_ord_least_squares)
y_pred_ridge = predict(X_test, w_ridge)
y_pred_weighted_ridge = predict(X_test, w_weighted_ridge)

rmse_ols = rmse(Y_test, y_pred_ord_least_squares)
rmse_ridge = rmse(Y_test, y_pred_ridge)
rmse_weighted_ridge = rmse(Y_test, y_pred_weighted_ridge)

print("\nRMSE Results:")
print(f"Ordinary Least Squares: {rmse_ols}")
print(f"Ridge Regression (λ=1.0): {rmse_ridge}")
print(f"Weighted Ridge Regression: {rmse_weighted_ridge}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(Y_test, y_pred_ord_least_squares, alpha=0.5, color='blue')
axes[0].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r', lw=1)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title(f'Ordinary Least Squares\nRMSE: {rmse_ols}')
axes[0].grid(True, alpha=0.2)

axes[1].scatter(Y_test, y_pred_ridge, alpha=0.5, color='green')
axes[1].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r', lw=1)
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title(f'Ridge Regression (λ=1.0)\nRMSE: {rmse_ridge}')
axes[1].grid(True, alpha=0.2)

axes[2].scatter(Y_test, y_pred_weighted_ridge, alpha=0.5, color='orange')
axes[2].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r', lw=1)
axes[2].set_xlabel('Actual')
axes[2].set_ylabel('Predicted')
axes[2].set_title(f'Weighted Ridge Regression\nRMSE: {rmse_weighted_ridge}')
axes[2].grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from q3_1 import gradient_descent_ridge
from q1_1 import rmse, data_matrix_bias, predict

# Write your code here ...
# Not autograded — function names and structure are flexible.

X_train_arr = pd.read_csv('X_train.csv').values
X_test_arr = pd.read_csv('X_test.csv').values
Y_train = pd.read_csv('y_train.csv').values
Y_test = pd.read_csv('y_test.csv').values

X_train = data_matrix_bias(X_train_arr)
X_test = data_matrix_bias(X_test_arr)

eta0 = 0.001
k = 0.001
T = 100
lamb = 1.0
schedules = ["constant", "exp_decay", "cosine"]

results = {}
losses_results = {}

for i, schedule in enumerate(schedules):
    print(f"{schedules[i]} learning rate schedule")
    
    w, losses = gradient_descent_ridge(X_train, Y_train, lamb=lamb, eta0=eta0, T=T,schedule=schedule, k_decay=k)
    
    y_pred = predict(X_test, w)
    test_rmse = rmse(Y_test, y_pred)
    
    results[schedule] = {
        'weights': w,
        'rmse': test_rmse,
        'name': schedules[i]
    }
    losses_results[schedule] = losses

print("RMSE Results Summary:")
for schedule in schedules:
    print(f"{results[schedule]['name']}: {results[schedule]['rmse']}")

# Losses comparison plot
plt.figure(figsize=(10, 10))
colors = ['blue', 'red', 'green']
for i, schedule in enumerate(schedules):
    plt.plot(range(T), losses_results[schedule], color=colors[i], linewidth=1, label=f'{results[schedule]["name"]} (RMSE: {results[schedule]["rmse"]})')

plt.title('Loss Comparison')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('results/ridge_gradient_descent_loss_comparison.png')
plt.show()
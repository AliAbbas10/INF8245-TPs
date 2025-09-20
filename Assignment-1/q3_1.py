import numpy as np


# Part (a)
def ridge_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb: float) -> np.ndarray:
    """
    Computes the gradient of Ridge regression loss.
    ∇L(w) = -2/n X^T (y - X w) + 2 λ w
    """
    # WRITE YOUR CODE HERE...
    n = X.shape[0]
    error = y - np.dot(X, w)
    gradient = (-2/n) * np.dot(X.T, error) + 2 * lamb * w
    return gradient


# Part (b)
def learning_rate_exp_decay(eta0: float, t: int, k_decay: float) -> float:
    # WRITE YOUR CODE HERE...
    return eta0 * np.exp(-k_decay * t)



# Part (c)
def learning_rate_cosine_annealing(eta0: float, t: int, T: int) -> float:
    # WRITE YOUR CODE HERE...
    return eta0 * 1/2 * (1 + np.cos(np.pi * t / T))


# Part (d)
def gradient_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb:float, eta: float) -> np.ndarray:
    # WRITE YOUR CODE HERE...
    gradient = ridge_gradient(X, y, w, lamb)
    return w - eta * gradient


# Part (e)
def gradient_descent_ridge(X, y, lamb=1.0, eta0=0.1, T=500, schedule="constant", k_decay=0.01):
    # WRITE YOUR CODE HERE...
    m = X.shape[1]
    w = np.zeros((m, 1))
    L = []

    # (Error form gradescope autograder fix), y should be a column vector
    y = y.reshape(-1, 1) if y.ndim == 1 or y.shape[1] != 1 else y
    
    for t in range(T):
        # (See report) for explanation of ridge regression loss = (1/n)||y - Xw||^2 + λ||w||^2
        n = X.shape[0]
        residuals = y - np.dot(X, w)
        mse_loss = (1/n) * np.sum(residuals**2)
        regularization = lamb * np.sum(w**2)
        loss = mse_loss + regularization
        L.append(loss)
        
        if schedule == "constant":
            eta = eta0
        elif schedule == "exp_decay":
            eta = learning_rate_exp_decay(eta0, t, k_decay)
        elif schedule == "cosine":
            eta = learning_rate_cosine_annealing(eta0, t, T)
        w = gradient_step(X, y, w, lamb, eta)

    return w.flatten(), L


# # Remove the following line if you are not using it:
# if __name__ == "__main__":

#     # If you want to test your functions, write your code here.
#     # If you write it outside this snippet, the autograder will fail!

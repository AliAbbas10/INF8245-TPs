import numpy as np
from scipy.stats import multivariate_normal
import typing


# -----------------------
# Gaussian Naive Bayes
# -----------------------
def gnb_fit_classifier(X: np.ndarray, Y: np.ndarray, smoothing: float = 1e-3) -> typing.Tuple:
    """
    Fits the GNB classifier on the training data
    """
    prior_probs = []
    means = []
    vars_ = []
    # Your implementation here
    classes = np.unique(Y)
    for k in classes:
        X_k = X[Y == k]
        
        # P(g=k)
        prior_k = len(X_k) / len(X)
        prior_probs.append(prior_k)
        
        #uk = E[X|g=k]
        mean_k = np.mean(X_k, axis=0)
        means.append(mean_k)
        
        # Class conditional variance
        var_k = np.var(X_k, axis=0) + smoothing
        vars_.append(var_k)

    return prior_probs, means, vars_


def gnb_predict(
    X: np.ndarray,
    prior_probs: typing.List[float],
    means: typing.List[np.ndarray],
    vars_: typing.List[np.ndarray],
    num_classes: int,
) -> np.ndarray:
    """
    Computes predictions from the GNB classifier
    """
    num_samples = X.shape[0]
    log_posteriors = np.zeros((num_samples, num_classes))
    
    for k in range(num_classes):
        log_prior_k = np.log(prior_probs[k])
    
        
        mean_k = means[k]
        var_k = vars_[k]
        
        diff_squared = (X - mean_k) ** 2
        normalized_diff = diff_squared / var_k
        
        # Log likelihood
        log_det_term = np.sum(np.log(var_k)) 
        quadratic_term = np.sum(normalized_diff, axis=1)
        log_likelihood_k = -0.5 * (log_det_term + quadratic_term)
        
        log_posteriors[:, k] = log_likelihood_k + log_prior_k
    
    preds = np.argmax(log_posteriors, axis=1)
    return preds


def gnb_classifier(train_set, train_labels, test_set, test_labels, smoothing=1e-3):
    """
    Runs GNB classifier and computes accuracy
    """
    num_classes = len(np.unique(train_labels))
    priors, means, vars_ = gnb_fit_classifier(train_set, train_labels, smoothing)
    y_pred = gnb_predict(test_set, priors, means, vars_, num_classes)
    accuracy = np.mean(y_pred == test_labels) * 100.0
    return accuracy


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    from data_process import preprocess_mnist_data
    from utils import visualize_image
    import matplotlib.pyplot as plt
    import numpy as np

    # MNIST dataset (from CSVs prepared by data_download.py)
    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist, _, _ = preprocess_mnist_data(
        "data/MNIST/mnist_train.csv", "data/MNIST/mnist_test.csv"
    )

    print("Evaluating on MNIST...")

    num_mnist_classes = len(np.unique(y_train_mnist))
    gnb_acc_mnist = gnb_classifier(X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist)
    class_priors, class_means, class_variances = gnb_fit_classifier(X_train_mnist, y_train_mnist)
    mnist_predictions = gnb_predict(X_test_mnist, class_priors, class_means, class_variances, num_mnist_classes)
    
    mnist_accuracy = np.mean(mnist_predictions == y_test_mnist) * 100.0
    print(f"MNIST - GNB accuracy: {gnb_acc_mnist:.2f} %")
    
    print("Perclass error rates for MNIST:")
    for digit_class in range(10):
        test_samples_mask = y_test_mnist == digit_class
        predicted_labels = mnist_predictions[test_samples_mask]
        true_labels = y_test_mnist[test_samples_mask]
        
        if len(true_labels) > 0:
            class_error_rate = (1 - np.mean(predicted_labels == true_labels)) * 100
    
    class_1_indices = np.where(y_test_mnist == 1)[0]
    correct_class_1 = class_1_indices[mnist_predictions[class_1_indices] == 1]

    if len(correct_class_1) > 0:
        idx = correct_class_1[0]
        visualize_image(X_test_mnist[idx], y_test_mnist[idx])
        
        # plt.figure(figsize=(5, 5))
        # image = X_test_mnist[idx].reshape(15, 15)
        # plt.imshow(image, cmap='gray')
        # plt.axis('off')
        # plt.savefig('class_1_example.png')
        # plt.close()

    # IRIS dataset (CSV created by data_download.py): last column is label
    train_iris = np.loadtxt("data/iris/iris_train.csv", delimiter=",")
    test_iris = np.loadtxt("data/iris/iris_test.csv", delimiter=",")
    X_train_iris, y_train_iris = train_iris[:, :-1], train_iris[:, -1].astype(int)
    X_test_iris, y_test_iris = test_iris[:, :-1], test_iris[:, -1].astype(int)

    print("\nEvaluating on IRIS...")
    gnb_acc_iris = gnb_classifier(X_train_iris, y_train_iris, X_test_iris, y_test_iris)
    print(f"IRIS - GNB accuracy: {gnb_acc_iris:.2f} %")


# # -----------------------
# # Quadratic Discriminant Analysis # You are lucky you don't have to do anything about this!
# # -----------------------
# def qda_fit_model(X: np.ndarray, Y: np.ndarray, reg: float = 1e-3) -> typing.Tuple:
#     """
#     Fit QDA model: compute mu_k and full covariance Sigma_k per class
#     """
#     priors, means, covariances = [], [], []
#     return priors, means, covariances


# def qda_predict(
#     X: np.ndarray, priors: typing.List[float], means: typing.List[np.ndarray], covariances: typing.List[np.ndarray]
# ) -> np.ndarray:
#     """
#     Computes predictions from a QDA classifier
#     """
#     log_probs = None
#     preds = None
#     return preds


# def qda_classifier(train_set, train_labels, test_set, test_labels, reg=1e-3):
#     """
#     Run QDA classifier and return accuracy
#     """
#     priors, means, covariances = qda_fit_model(train_set, train_labels, reg)
#     y_pred = qda_predict(test_set, priors, means, covariances)
#     accuracy = np.mean(y_pred == test_labels) * 100.0
#     return accuracy

import numpy as np


class KDNode:
    """A node in a k-d tree."""

    def __init__(self, data, label, dim, left=None, right=None):
        """
        data: The feature vector of the point at this node.
        label: The class label of the point.
        dim: The dimension used for splitting at this node.
        left: The left child node.
        right: The right child node.
        """
        self.data = data
        self.label = label
        self.dim = dim
        self.left = left
        self.right = right


class KDTree:
    """A k-d tree data structure for efficient nearest neighbor search."""

    def __init__(self, X_train, y_train):
        """
        X_train: The training features.
        y_train: The training labels.

        Attributes:
            - dims: Number of dimensions in the feature space (will be needed for building the tree)
            - root: The root node of the k-d tree
        """
        self.dims = X_train.shape[1]
        self.root = self._build_tree(X_train, y_train, depth=0)

    def _build_tree(self, X_train, y_train, depth):
        """
        Recursively builds the k-d tree.
        X_train: The subset of training features for this node.
        y_train: The corresponding labels.
        depth: The current recursion depth.
        """
        if len(X_train) == 0:
            return None

        # TODO : Implement the k-d tree building logic

        axis = depth % self.dims
        sorted_indices = np.argsort(X_train[:, axis])
        X_sorted = X_train[sorted_indices]
        y_sorted = y_train[sorted_indices]
        median = len(X_sorted) // 2
        root = KDNode(data=X_sorted[median], label=y_sorted[median], dim=axis)
        root.left = self._build_tree(X_sorted[:median], y_sorted[:median], depth + 1)
        root.right = self._build_tree(X_sorted[median + 1:], y_sorted[median + 1:], depth + 1)
        return root

    def _find_nearest(self, node, query_point, best_guess=None, best_dist=np.inf):
        """
        Recursively finds the nearest neighbor to a query point.
        node: The current node in the tree.
        query_point: The point for which to find the nearest neighbor.
        best_guess: The current best neighbor found so far.
        best_dist: The distance to the current best guess.

        Returns:
            best_guess: The nearest neighbor found.
            best_dist: The distance to the nearest neighbor.
        """
        if node is None:
            return best_guess, best_dist

        # TODO: Implement the nearest neighbor search logic
        point = node.data
        distance = np.linalg.norm(point - query_point)
        if distance < best_dist:
            best_dist = distance
            best_guess = node
        axis = node.dim
        difference = query_point[axis] - point[axis]

        if distance == 0:
            return node, 0
    
        if distance < best_dist:
            best_dist = distance
            best_guess = node

        axis = node.dim
        difference = query_point[axis] - point[axis]
        
        if difference <= 0:
            first_child_to_search, second_child = node.left, node.right
        else:
            first_child_to_search, second_child = node.right, node.left

        best_guess, best_dist = self._find_nearest(first_child_to_search, query_point, best_guess, best_dist)

        if abs(difference) < best_dist:
            best_guess, best_dist = self._find_nearest(second_child, query_point, best_guess, best_dist)

        return best_guess, best_dist

    def find_nearest_neighbor(self, query_point):
        """
        Public method to find the nearest neighbor.
        query_point: The point for which to find the nearest neighbor.
        """
        return self._find_nearest(self.root, query_point)[0]


def kdtree_1nn_classifier(X_train, y_train, X_test):
    """
    Classifies a set of test points using a 1-NN search with a k-d tree.
    X_train: The training features. np.ndarray of shape (num_train_samples, num_features)
    y_train: The training labels. np.ndarray of shape (num_train_samples,)
    X_test: The test features. np.ndarray of shape (num_test_samples, num_features)

    Returns:
        predictions: The predicted labels for the test set. np.ndarray of shape (num_test_samples,)
    """
    predictions = []
    tree = KDTree(X_train, y_train)
    for query_point in X_test:
        nearest = tree.find_nearest_neighbor(query_point)
        if hasattr(nearest, "label"):
            predictions.append(nearest.label)
        else:
            predictions.append(nearest)
    return np.array(predictions)


if __name__ == "__main__":
    # Example usage
    from data_process import preprocess_mnist_data, preprocess_credit_card

    # Load and preprocess the MNIST dataset
    # X_train, y_train, X_test, y_test, mean, std = preprocess_mnist_data("data/MNIST/train.csv", "data/MNIST/t10k.csv")
    X_train, y_train, X_test, y_test, mean, std = preprocess_credit_card(
        "Assignment-2/data/credit_card_fraud/credit_card_fraud_train.csv", "Assignment-2/data/credit_card_fraud/credit_card_fraud_test.csv"
    )
    print("Data loaded and preprocessed.")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    # Use a subset for quick testing (you MUST remove this in real use)
    X_train_small = X_train[:1000]
    y_train_small = y_train[:1000]
    X_test_small = X_test[:100]
    y_test_small = y_test[:100]

    # Classify using k-d tree 1-NN
    predictions = kdtree_1nn_classifier(X_train, y_train, X_test_small)

    # Print results
    print("Predictions:", predictions)
    print("True labels:", y_test_small)

    accuracy = np.mean(predictions == y_test_small)
    print(f"Accuracy: {accuracy * 100:.2f}%")

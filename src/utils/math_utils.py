import numpy as np

def sigmoid(x):
    """Compute the sigmoid function."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Compute the derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    """Compute the softmax function."""
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    else:
        # For 2D array: (output_size, batch_size)
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

def binary_cross_entropy(y_true, y_pred):
    """Compute the binary cross-entropy loss."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    """
    Compute the categorical cross-entropy loss.

    Args:
        y_true: True labels (one-hot encoded) - shape (num_classes, batch_size)
        y_pred: Predicted probabilities - shape (num_classes, batch_size)

    Returns:
        float: Average loss
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

def accuracy(y_true, y_pred):
    """
    Compute accuracy for classification.

    Args:
        y_true: True labels - can be (batch_size,) or (num_classes, batch_size)
        y_pred: Predicted probabilities - can be (batch_size,) or (num_classes, batch_size)

    Returns:
        float: Accuracy
    """
    # If one-hot encoded, convert to class indices
    if y_true.ndim == 2 and y_true.shape[0] > 1:
        y_true = np.argmax(y_true, axis=0)
    if y_pred.ndim == 2 and y_pred.shape[0] > 1:
        y_pred = np.argmax(y_pred, axis=0)
    elif y_pred.ndim == 1:
        # Binary classification with sigmoid
        y_pred = (y_pred > 0.5).astype(int)

    return np.mean(y_pred == y_true)
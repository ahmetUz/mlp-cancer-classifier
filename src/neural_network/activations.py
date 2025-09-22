import numpy as np
from ..utils.math_utils import sigmoid, sigmoid_derivative, softmax


class ActivationFunction:
    """Base class for activation functions"""

    def forward(self, x):
        """Forward pass"""
        raise NotImplementedError

    def backward(self, x):
        """Backward pass - compute derivative"""
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""

    def forward(self, x):
        """Apply sigmoid function"""
        return sigmoid(x)

    def backward(self, x):
        """Compute sigmoid derivative"""
        return sigmoid_derivative(x)


class Softmax(ActivationFunction):
    """Softmax activation function for output layer"""

    def forward(self, x):
        """Apply softmax function"""
        return softmax(x)

    def backward(self, x):
        """Compute softmax derivative"""
        # Pour la rétropropagation avec cross-entropy,
        # la dérivée sera calculée directement dans la loss
        s = softmax(x)
        # Jacobienne de softmax est complexe, on la calcule si nécessaire
        if x.ndim == 1:
            return s * (1 - s)
        else:
            # Pour un batch, retourne la diagonale de la jacobienne
            return s * (1 - s)


class ReLU(ActivationFunction):
    """ReLU activation function"""

    def forward(self, x):
        """Apply ReLU function"""
        return np.maximum(0, x)

    def backward(self, x):
        """Compute ReLU derivative"""
        return (x > 0).astype(float)


class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function"""

    def forward(self, x):
        """Apply tanh function"""
        return np.tanh(x)

    def backward(self, x):
        """Compute tanh derivative"""
        return 1 - np.tanh(x) ** 2


def get_activation(activation_name):
    """Factory function to get activation by name"""
    activations = {
        'sigmoid': Sigmoid(),
        'softmax': Softmax(),
        'relu': ReLU(),
        'tanh': Tanh()
    }

    if activation_name.lower() in activations:
        return activations[activation_name.lower()]
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")
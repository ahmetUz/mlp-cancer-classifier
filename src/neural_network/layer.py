import numpy as np
from .activations import get_activation


class Layer:
    """Dense (fully connected) layer for neural network"""

    def __init__(self, input_size, output_size, activation='sigmoid', weights_initializer='he',
                 l2_lambda=0.0, dropout_rate=0.0):
        """
        Initialize a dense layer

        Args:
            input_size (int): Number of input neurons
            output_size (int): Number of output neurons
            activation (str): Activation function name
            weights_initializer (str): Weight initialization strategy
            l2_lambda (float): L2 regularization coefficient (0 = no regularization)
            dropout_rate (float): Dropout rate (0 = no dropout, must be < 1)
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError("Dropout rate must be >= 0 and < 1")

        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = get_activation(activation)
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.training = True

        # Initialize weights and biases
        self.weights = self._initialize_weights(weights_initializer)
        self.biases = np.zeros((output_size, 1))

        # Store values for backpropagation
        self.last_input = None
        self.last_z = None  # Linear output (before activation)
        self.last_output = None  # Output after activation
        self.dropout_mask = None  # Mask for dropout

    def _initialize_weights(self, method):
        """Initialize weights using specified method"""
        if method.lower() == 'random':
            return np.random.randn(self.output_size, self.input_size) * 0.1
        elif method.lower() == 'xavier':
            # Xavier/Glorot initialization
            limit = np.sqrt(6.0 / (self.input_size + self.output_size))
            return np.random.uniform(-limit, limit, (self.output_size, self.input_size))
        elif method.lower() == 'he' or method.lower() == 'heuniform':
            # He initialization (good for ReLU)
            std = np.sqrt(2.0 / self.input_size)
            return np.random.randn(self.output_size, self.input_size) * std
        else:
            # Default: small random values
            return np.random.randn(self.output_size, self.input_size) * 0.01

    def forward(self, inputs):
        """
        Forward propagation through the layer

        Args:
            inputs (np.array): Input data of shape (input_size, batch_size) or (input_size,)

        Returns:
            np.array: Output after applying weights, biases, activation, and dropout
        """
        # Ensure input is 2D (input_size, batch_size)
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)

        # Store input for backpropagation
        self.last_input = inputs

        # Linear transformation: z = W*x + b
        self.last_z = np.dot(self.weights, inputs) + self.biases

        # Apply activation function
        self.last_output = self.activation_func.forward(self.last_z)

        # Apply dropout (only during training)
        if self.training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.last_output.shape)
            self.last_output = (self.last_output * self.dropout_mask) / (1 - self.dropout_rate)

        return self.last_output

    def backward(self, grad_output, learning_rate, is_output_layer=False):
        """
        Backward propagation through the layer

        Args:
            grad_output (np.array): Gradient of loss w.r.t layer output
            learning_rate (float): Learning rate for weight updates
            is_output_layer (bool): If True, skip activation gradient (already included in grad_output)

        Returns:
            np.array: Gradient of loss w.r.t layer input
        """
        # Ensure grad_output is 2D
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)

        # Apply dropout mask to gradient (same mask as forward pass)
        if self.training and self.dropout_rate > 0:
            grad_output = (grad_output * self.dropout_mask) / (1 - self.dropout_rate)

        batch_size = self.last_input.shape[1]

        # Compute gradient w.r.t activation input (z)
        # For output layer with cross-entropy: grad_output is already dL/dz (y_pred - y)
        # For hidden layers: dL/dz = dL/da * da/dz
        if is_output_layer:
            grad_z = grad_output  # Already dL/dz for cross-entropy
        else:
            activation_grad = self.activation_func.backward(self.last_z)
            grad_z = grad_output * activation_grad

        # Compute gradients w.r.t weights and biases
        grad_weights = np.dot(grad_z, self.last_input.T) / batch_size
        grad_biases = np.mean(grad_z, axis=1, keepdims=True)

        # Add L2 regularization gradient: d/dW (lambda/2 * ||W||^2) = lambda * W
        if self.l2_lambda > 0:
            grad_weights += self.l2_lambda * self.weights

        # Compute gradient w.r.t input
        grad_input = np.dot(self.weights.T, grad_z)

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

    def l2_penalty(self):
        """Calculate L2 regularization penalty for this layer"""
        if self.l2_lambda > 0:
            return (self.l2_lambda / 2) * np.sum(self.weights ** 2)
        return 0.0

    def get_params(self):
        """Get layer parameters for saving"""
        return {
            'weights': self.weights,
            'biases': self.biases,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation_func.__class__.__name__.lower(),
            'l2_lambda': self.l2_lambda,
            'dropout_rate': self.dropout_rate
        }

    def set_params(self, params):
        """Set layer parameters for loading"""
        self.weights = params['weights']
        self.biases = params['biases']
        self.activation_func = get_activation(params['activation'])
        self.l2_lambda = params.get('l2_lambda', 0.0)
        self.dropout_rate = params.get('dropout_rate', 0.0)
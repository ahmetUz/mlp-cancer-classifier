import numpy as np
from .activations import get_activation


class Dropout:
    """Dropout layer for regularization"""

    def __init__(self, rate=0.5):
        """
        Initialize dropout layer

        Args:
            rate (float): Fraction of neurons to drop (0 to 1)
        """
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, inputs):
        """
        Forward pass with dropout

        Args:
            inputs (np.array): Input data

        Returns:
            np.array: Output with dropout applied (if training)
        """
        if self.training and self.rate > 0:
            # Create binary mask (1 = keep, 0 = drop)
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape)
            # Apply mask and scale by 1/(1-rate) to maintain expected values
            return (inputs * self.mask) / (1 - self.rate)
        else:
            # During inference, don't apply dropout
            return inputs

    def backward(self, grad_output, learning_rate=None):
        """
        Backward pass through dropout

        Args:
            grad_output (np.array): Gradient from next layer
            learning_rate: Unused, for API compatibility

        Returns:
            np.array: Gradient with same mask applied
        """
        if self.training and self.rate > 0:
            return (grad_output * self.mask) / (1 - self.rate)
        return grad_output

    def get_params(self):
        """Get layer parameters for saving"""
        return {
            'type': 'dropout',
            'rate': self.rate
        }


class Layer:
    """Dense (fully connected) layer for neural network"""

    def __init__(self, input_size, output_size, activation='sigmoid', weights_initializer='he', l2_lambda=0.0):
        """
        Initialize a dense layer

        Args:
            input_size (int): Number of input neurons
            output_size (int): Number of output neurons
            activation (str): Activation function name
            weights_initializer (str): Weight initialization strategy
            l2_lambda (float): L2 regularization coefficient (0 = no regularization)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = get_activation(activation)
        self.l2_lambda = l2_lambda

        # Initialize weights and biases
        self.weights = self._initialize_weights(weights_initializer)
        self.biases = np.zeros((output_size, 1))

        # Store values for backpropagation
        self.last_input = None
        self.last_z = None  # Linear output (before activation)
        self.last_output = None  # Output after activation

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
            np.array: Output after applying weights, biases and activation
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

        return self.last_output

    def backward(self, grad_output, learning_rate):
        """
        Backward propagation through the layer

        Args:
            grad_output (np.array): Gradient of loss w.r.t layer output
            learning_rate (float): Learning rate for weight updates

        Returns:
            np.array: Gradient of loss w.r.t layer input
        """
        # Ensure grad_output is 2D
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)

        batch_size = self.last_input.shape[1]

        # Compute gradient w.r.t activation input (z)
        # For most activations: dL/dz = dL/da * da/dz
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
            'l2_lambda': self.l2_lambda
        }

    def set_params(self, params):
        """Set layer parameters for loading"""
        self.weights = params['weights']
        self.biases = params['biases']
        self.activation_func = get_activation(params['activation'])
        self.l2_lambda = params.get('l2_lambda', 0.0)
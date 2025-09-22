import numpy as np
from .activations import get_activation


class Layer:
    """Dense (fully connected) layer for neural network"""

    def __init__(self, input_size, output_size, activation='sigmoid', weights_initializer='random'):
        """
        Initialize a dense layer

        Args:
            input_size (int): Number of input neurons
            output_size (int): Number of output neurons
            activation (str): Activation function name
            weights_initializer (str): Weight initialization strategy
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = get_activation(activation)

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

        # Compute gradient w.r.t input
        grad_input = np.dot(self.weights.T, grad_z)

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

    def backward_with_gradients(self, grad_output):
        """
        Backward pass that returns gradients without updating weights
        Used for gradient accumulation in batch processing

        Args:
            grad_output (np.array): Gradient of loss w.r.t layer output

        Returns:
            tuple: (grad_input, grad_weights, grad_biases)
        """
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)

        batch_size = self.last_input.shape[1]

        # Compute gradient w.r.t activation input
        activation_grad = self.activation_func.backward(self.last_z)
        grad_z = grad_output * activation_grad

        # Compute gradients
        grad_weights = np.dot(grad_z, self.last_input.T) / batch_size
        grad_biases = np.mean(grad_z, axis=1, keepdims=True)
        grad_input = np.dot(self.weights.T, grad_z)

        return grad_input, grad_weights, grad_biases

    def update_weights(self, grad_weights, grad_biases, learning_rate):
        """Update weights and biases with given gradients"""
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

    def get_params(self):
        """Get layer parameters for saving"""
        return {
            'weights': self.weights,
            'biases': self.biases,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation_func.__class__.__name__.lower()
        }

    def set_params(self, params):
        """Set layer parameters for loading"""
        self.weights = params['weights']
        self.biases = params['biases']
        self.activation_func = get_activation(params['activation'])
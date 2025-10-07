import numpy as np
import pickle
import os
from .layer import Layer
from ..utils.math_utils import binary_cross_entropy, categorical_cross_entropy, accuracy


class Network:
    """Multi-layer perceptron neural network"""

    def __init__(self):
        """Initialize an empty network"""
        self.layers = []
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def add_layer(self, layer):
        """
        Add a layer to the network

        Args:
            layer (Layer): Layer object to add
        """
        self.layers.append(layer)

    def create_network(self, layer_configs):
        """
        Create network from configuration list

        Args:
            layer_configs (list): List of dicts with layer parameters
                Example: [
                    {'input_size': 30, 'output_size': 24, 'activation': 'sigmoid', 'weights_initializer': 'random'},
                    {'input_size': 24, 'output_size': 24, 'activation': 'sigmoid', 'weights_initializer': 'random'},
                    {'input_size': 24, 'output_size': 1, 'activation': 'sigmoid', 'weights_initializer': 'random'}
                ]
        """
        self.layers = []
        for config in layer_configs:
            layer = Layer(
                input_size=config['input_size'],
                output_size=config['output_size'],
                activation=config.get('activation', 'sigmoid'),
                weights_initializer=config.get('weights_initializer', 'random')
            )
            self.add_layer(layer)

    def forward(self, X):
        """
        Forward propagation through entire network

        Args:
            X (np.array): Input data of shape (features, samples) or (features) for single sample

        Returns:
            np.array: Network output
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Propagate through all layers
        output = X
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, X, y, learning_rate):
        """
        Backward propagation through entire network

        Args:
            X (np.array): Input data
            y (np.array): True labels (one-hot encoded for categorical)
            learning_rate (float): Learning rate

        Returns:
            float: Loss value
        """
        # Forward pass
        y_pred = self.forward(X)

        # Ensure y and y_pred have same shape
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)

        # Determine which loss function to use based on output shape
        output_layer = self.layers[-1]
        is_softmax = output_layer.activation_func.__class__.__name__ == 'Softmax'

        # Calculate loss
        if is_softmax and y_pred.shape[0] > 1:
            # Categorical cross-entropy for softmax
            loss = categorical_cross_entropy(y, y_pred)
        else:
            # Binary cross-entropy for sigmoid
            loss = binary_cross_entropy(y, y_pred)

        # Calculate initial gradient for backpropagation
        # For cross-entropy with softmax/sigmoid: gradient simplifies to (y_pred - y)
        grad_output = y_pred - y

        # Backward pass through all layers (in reverse order)
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)

        return loss

    def train_batch(self, X_batch, y_batch, learning_rate):
        """
        Train on a single batch

        Args:
            X_batch (np.array): Batch of input data
            y_batch (np.array): Batch of labels
            learning_rate (float): Learning rate

        Returns:
            tuple: (loss, accuracy) for this batch
        """
        # Forward and backward pass
        loss = self.backward(X_batch.T, y_batch.T, learning_rate)

        # Calculate accuracy
        y_pred = self.forward(X_batch.T)
        acc = accuracy(y_batch.T, y_pred)

        return loss, acc

    def train(self, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.01,
              batch_size=32, verbose=True):
        """
        Train the network

        Args:
            X_train (np.array): Training data
            y_train (np.array): Training labels
            X_val (np.array): Validation data
            y_val (np.array): Validation labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            verbose (bool): Print training progress
        """
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_train_losses = []
            epoch_train_accs = []

            # Train on batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_loss, batch_acc = self.train_batch(X_batch, y_batch, learning_rate)
                epoch_train_losses.append(batch_loss)
                epoch_train_accs.append(batch_acc)

            # Calculate epoch metrics
            train_loss = np.mean(epoch_train_losses)
            train_acc = np.mean(epoch_train_accs)

            # Validation metrics
            val_loss, val_acc = self.evaluate(X_val, y_val)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print progress
            if verbose:
                print(f"epoch {epoch + 1:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

    def evaluate(self, X, y):
        """
        Evaluate network on given data

        Args:
            X (np.array): Input data
            y (np.array): True labels (can be one-hot or single values)

        Returns:
            tuple: (loss, accuracy)
        """
        y_pred_raw = self.forward(X.T)

        # Determine which loss to use
        output_layer = self.layers[-1]
        is_softmax = output_layer.activation_func.__class__.__name__ == 'Softmax'

        # Prepare labels for loss computation
        if y.ndim == 1:
            # Convert to one-hot if needed for softmax
            if is_softmax and y_pred_raw.shape[0] > 1:
                y_one_hot = np.zeros((y_pred_raw.shape[0], len(y)))
                y_one_hot[y.astype(int), np.arange(len(y))] = 1
                loss = categorical_cross_entropy(y_one_hot, y_pred_raw)
            else:
                y_reshaped = y.reshape(1, -1)
                loss = binary_cross_entropy(y_reshaped, y_pred_raw)
        else:
            # Already one-hot encoded
            if is_softmax and y_pred_raw.shape[0] > 1:
                loss = categorical_cross_entropy(y.T, y_pred_raw)
            else:
                loss = binary_cross_entropy(y, y_pred_raw.flatten())

        # For accuracy, convert predictions
        if is_softmax and y_pred_raw.shape[0] > 1:
            y_pred_classes = np.argmax(y_pred_raw, axis=0)
        else:
            y_pred_classes = (y_pred_raw.flatten() > 0.5).astype(int)

        y_true_classes = y if y.ndim == 1 else np.argmax(y, axis=1) if y.shape[1] > 1 else y.flatten()
        acc = np.mean(y_pred_classes == y_true_classes)

        return loss, acc

    def predict(self, X):
        """
        Make predictions on input data

        Args:
            X (np.array): Input data

        Returns:
            np.array: Predictions
        """
        output = self.forward(X.T)
        return output.flatten() if output.shape[1] == 1 else output.T

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X (np.array): Input data

        Returns:
            np.array: Class probabilities
        """
        return self.predict(X)

    def predict_classes(self, X, threshold=0.5):
        """
        Predict binary classes

        Args:
            X (np.array): Input data
            threshold (float): Decision threshold

        Returns:
            np.array: Predicted classes (0 or 1)
        """
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

    def save(self, filepath):
        """
        Save the trained model

        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'layers': [layer.get_params() for layer in self.layers],
            'history': self.history
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    def load(self, filepath):
        """
        Load a trained model

        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Reconstruct layers
        self.layers = []
        layer_configs = model_data['layers']

        for config in layer_configs:
            layer = Layer(
                input_size=config['input_size'],
                output_size=config['output_size'],
                activation=config['activation']
            )
            layer.set_params(config)
            self.add_layer(layer)

        # Restore history if available
        if 'history' in model_data:
            self.history = model_data['history']

        print(f"Model loaded from: {filepath}")

    def get_architecture(self):
        """
        Get network architecture summary

        Returns:
            str: Architecture description
        """
        architecture = "Network Architecture:\n"
        architecture += "=" * 50 + "\n"

        for i, layer in enumerate(self.layers):
            architecture += f"Layer {i + 1}: {layer.input_size} -> {layer.output_size} "
            architecture += f"({layer.activation_func.__class__.__name__})\n"

        total_params = sum(layer.weights.size + layer.biases.size for layer in self.layers)
        architecture += f"\nTotal parameters: {total_params}"

        return architecture
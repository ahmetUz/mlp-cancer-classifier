#!/usr/bin/env python
"""
Training script for the MLP classifier.

Usage:
    python train.py                                    # Uses data/data_train.csv and data/data_val.csv with default [64, 32] layers
    python train.py data/data_training.csv             # Uses evaluation.py format (no header) with default layers
    python train.py --layers 128 64 32                 # Uses custom hidden layers [128, 64, 32]
    python train.py data/data_training.csv -l 32 16    # Uses custom file and layers
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neural_network.network import Network


def load_data_with_header(train_path, val_path):
    """Load data from CSV with header (split_dataset.py format)"""
    import pandas as pd

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.drop(['diagnosis'], axis=1).values
    y_train = train_df['diagnosis'].values

    X_val = val_df.drop(['diagnosis'], axis=1).values
    y_val = val_df['diagnosis'].values

    return X_train, y_train, X_val, y_val


def load_data_no_header(path):
    """Load data from CSV without header (evaluation.py format)"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            diagnosis = 1 if parts[1] == 'M' else 0
            features = [float(x) for x in parts[2:]]
            data.append((features, diagnosis))

    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    return X, y


def standardize_features(X_train, X_val=None):
    """Standardize features (zero mean, unit variance)"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std if X_val is not None else None

    return X_train_norm, X_val_norm, mean, std


def to_onehot(y, num_classes=2):
    """Convert labels to one-hot encoding"""
    onehot = np.zeros((len(y), num_classes))
    onehot[np.arange(len(y)), y.astype(int)] = 1
    return onehot


def plot_learning_curves(history):
    """Plot training and validation curves"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-', label='Training loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Training accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/learning_curves.png', dpi=100, bbox_inches='tight')
    print("Learning curves saved to: models/learning_curves.png")
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train MLP classifier for cancer diagnosis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                                    # Default layers [64, 32]
  python train.py --layers 128 64 32                 # Custom hidden layers
  python train.py data/custom.csv -l 32 16           # Custom file and layers
        """
    )
    parser.add_argument(
        'data_file',
        nargs='?',
        default=None,
        help='Optional: path to training data file (evaluation.py format, no header)'
    )
    parser.add_argument(
        '--layers', '-l',
        type=int,
        nargs='*',
        default=[64, 32],
        metavar='SIZE',
        help='Hidden layer sizes (default: 64 32). Must specify at least 2 layers.'
    )

    args = parser.parse_args()

    # Validate layers argument
    if len(args.layers) < 2:
        parser.error("--layers must specify at least 2 hidden layers")

    if any(size <= 0 for size in args.layers):
        parser.error("--layers must contain only positive integers (greater than 0)")

    # Display network configuration
    print("=" * 60)
    print("Network Configuration")
    print("=" * 60)
    print(f"Hidden layers: {args.layers}")
    print("=" * 60)
    print()

    # Determine which mode to use
    if args.data_file:
        # Evaluation mode: single file without header
        train_path = args.data_file
        print(f"Loading data from {train_path} (evaluation mode)...")

        X_all, y_all = load_data_no_header(train_path)

        # Split 90/10 for train/val
        n_samples = len(X_all)
        indices = np.random.permutation(n_samples)
        split_idx = int(0.9 * n_samples)

        X_train, y_train = X_all[indices[:split_idx]], y_all[indices[:split_idx]]
        X_val, y_val = X_all[indices[split_idx:]], y_all[indices[split_idx:]]

        # Standardize
        X_train, X_val, mean, std = standardize_features(X_train, X_val)

        # Save normalization params for prediction
        np.save('models/normalization_params.npy', {'mean': mean, 'std': std})

    else:
        # Standard mode: two files with header
        train_path = 'data/data_train.csv'
        val_path = 'data/data_val.csv'

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print("Error: Training or validation data not found!")
            print("Run: python scripts/split_dataset.py data/data.csv")
            print("Or:  python scripts/train.py data/data_training.csv")
            sys.exit(1)

        print("Loading data (standard mode)...")
        X_train, y_train, X_val, y_val = load_data_with_header(train_path, val_path)

        # Standardize
        X_train, X_val, mean, std = standardize_features(X_train, X_val)
        np.save('models/normalization_params.npy', {'mean': mean, 'std': std})

    print(f"x_train shape : ({X_train.shape[0]}, {X_train.shape[1]})")
    print(f"x_valid shape : ({X_val.shape[0]}, {X_val.shape[1]})")

    # Create network with configurable layers
    n_features = X_train.shape[1]
    network = Network()

    # Build layer configs dynamically from args.layers
    layer_configs = []

    # Hidden layers
    prev_size = n_features
    for layer_size in args.layers:
        layer_configs.append({
            'input_size': prev_size,
            'output_size': layer_size,
            'activation': 'relu',
            'l2_lambda': 0.0005
        })
        prev_size = layer_size

    # Output layer
    layer_configs.append({
        'input_size': prev_size,
        'output_size': 2,
        'activation': 'softmax'
    })

    network.create_network(layer_configs)

    # Convert labels to one-hot
    y_train_onehot = to_onehot(y_train)
    y_val_onehot = to_onehot(y_val)

    # Train
    network.train(
        X_train, y_train_onehot,
        X_val, y_val_onehot,
        epochs=600,
        learning_rate=0.08,
        batch_size=16,
        patience=60,
        verbose=True
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    train_loss, train_acc = network.evaluate(X_train, y_train_onehot)
    val_loss, val_acc = network.evaluate(X_val, y_val_onehot)

    print(f"\nFinal Results:")
    print(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    network.save('models/trained_model.pkl')

    # Plot learning curves
    print("\nGenerating learning curves...")
    plot_learning_curves(network.history)


if __name__ == "__main__":
    main()

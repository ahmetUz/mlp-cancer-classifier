#!/usr/bin/env python
"""
Training script for the MLP classifier.

Auto-detects CSV format (with header vs without header) by reading the first line.

Usage:
    python train.py                                    # Uses data/data_train.csv and data/data_val.csv with default [64, 32] layers
    python train.py data/data_train.csv                # Auto-detects header format, uses with validation file
    python train.py --layers 128 64 32                 # Uses custom hidden layers [128, 64, 32]
    python train.py data/data_train.csv -l 32 16       # Auto-detects format, uses custom file and layers
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neural_network.network import Network
from src.data.loaders import detect_csv_format, load_train_val_data, load_data_no_header
from src.data.preprocessing import standardize_features, to_onehot
from src.data.visualization import plot_learning_curves


def dropout_rate(value):
    """
    Custom argparse type validator for dropout rate.

    Args:
        value: String value from command line

    Returns:
        float: Validated dropout rate

    Raises:
        argparse.ArgumentTypeError: If value is not in valid range [0.0, 1.0)
    """
    try:
        rate = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: '{value}'")

    if not (0.0 <= rate < 1.0):
        raise argparse.ArgumentTypeError(f"dropout rate must be >= 0.0 and < 1.0 (got {rate})")

    return rate


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
  python train.py --dropout 0.3                      # Add 30% dropout between hidden layers
  python train.py -l 128 64 32 -d 0.5                # Custom layers with 50% dropout
        """
    )
    parser.add_argument(
        'data_file',
        nargs='?',
        default=None,
        help='Optional: path to training data file (auto-detects CSV format with or without header)'
    )
    parser.add_argument(
        '--layers', '-l',
        type=int,
        nargs='*',
        default=[256, 256, 256, 256],
        metavar='SIZE',
        help='Hidden layer sizes (default: 64 32). Must specify at least 2 layers.'
    )
    parser.add_argument(
        '--dropout', '-d',
        type=dropout_rate,
        default=0.0,

        metavar='RATE',
        help='Dropout rate between hidden layers (default: 0.0, range: [0.0, 1.0))'
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
    if args.dropout > 0:
        print(f"Dropout rate: {args.dropout}")
    print("=" * 60)
    print()

    # Determine which mode to use
    if args.data_file:
        # Single file mode: auto-detect format
        train_path = args.data_file

        if not os.path.exists(train_path):
            print(f"Error: File not found: {train_path}")
            sys.exit(1)

        # Auto-detect CSV format
        try:
            csv_format = detect_csv_format(train_path)
            print(f"Detected CSV format: {csv_format}")
        except (FileNotFoundError, PermissionError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

        if csv_format == 'with_header':
            # File has header - need to check if we have a validation file
            # Assume validation file is in same directory with _val suffix
            base_path = train_path.replace('_train.csv', '')
            val_path = base_path + '_val.csv'

            if os.path.exists(val_path):
                # Use separate train/val files
                print(f"Loading data from {train_path} and {val_path}...")
                X_train, y_train, X_val, y_val = load_train_val_data(train_path, val_path)
            else:
                print(f"Error: Validation file not found: {val_path}")
                print("Please run split_dataset.py first to create train/val split")
                sys.exit(1)
        else:  # no_header
            # Look for corresponding test file (data_training.csv -> data_test.csv)
            base_path = train_path.replace('_training.csv', '')
            test_path = base_path + '_test.csv'

            if os.path.exists(test_path):
                print(f"Loading data from {train_path} and {test_path}...")
                X_train, y_train = load_data_no_header(train_path)
                X_val, y_val = load_data_no_header(test_path)
            else:
                print(f"Error: Test file not found: {test_path}")
                print("Expected files: data_training.csv and data_test.csv")
                sys.exit(1)

        # Standardize
        X_train, X_val, mean, std = standardize_features(X_train, X_val)

        # Save normalization params for prediction
        np.save('models/normalization_params.npy', {'mean': mean, 'std': std})

    else:
        # Standard mode: use default two files with header
        train_path = 'data/data_train.csv'
        val_path = 'data/data_val.csv'

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print("Error: Training or validation data not found!")
            print("Run: python scripts/split_dataset.py data/data.csv")
            print("Or:  python scripts/train.py data/data_train.csv")
            sys.exit(1)

        print("Loading data (standard mode)...")
        X_train, y_train, X_val, y_val = load_train_val_data(train_path, val_path)

        # Standardize
        X_train, X_val, mean, std = standardize_features(X_train, X_val)
        np.save('models/normalization_params.npy', {'mean': mean, 'std': std})

    print(f"x_train shape : ({X_train.shape[0]}, {X_train.shape[1]})")
    print(f"x_valid shape : ({X_val.shape[0]}, {X_val.shape[1]})")

    # ===========================================
    # CONFIGURATION: Choisir l'activation de sortie
    # ===========================================
    # "softmax" : 2 neurones, categorical cross-entropy, labels one-hot
    # "sigmoid" : 1 neurone, binary cross-entropy, labels simples (0/1)
    OUTPUT_ACTIVATION = "softmax"
    # ===========================================

    # Create network with configurable layers
    n_features = X_train.shape[1]
    network = Network()

    # Build layer configs dynamically from args.layers
    layer_configs = []

    # Hidden layers (with optional dropout)
    prev_size = n_features
    for layer_size in args.layers:
        layer_configs.append({
            'input_size': prev_size,
            'output_size': layer_size,
            'activation': 'relu',
            'l2_lambda': 0.000005,
            'dropout_rate': args.dropout
        })
        prev_size = layer_size

    # Output layer (no dropout)
    if OUTPUT_ACTIVATION == "sigmoid":
        layer_configs.append({
            'input_size': prev_size,
            'output_size': 1,
            'activation': 'sigmoid'
        })
    else:
        layer_configs.append({
            'input_size': prev_size,
            'output_size': 2,
            'activation': 'softmax'
        })

    network.create_network(layer_configs)

    # Prepare labels selon l'activation choisie
    if OUTPUT_ACTIVATION == "sigmoid":
        y_train_prepared = y_train
        y_val_prepared = y_val
    else:
        y_train_prepared = to_onehot(y_train)
        y_val_prepared = to_onehot(y_val)

    # Train
    network.train(
        X_train, y_train_prepared,
        X_val, y_val_prepared,
        # === PARAMETRES NORMAUX ===
        epochs=1500,
        learning_rate=0.0007,
        batch_size=64,
        patience=500,
        # === PARAMETRES OVERFIT ===
        # epochs=500,
        # learning_rate=0.18,
        # batch_size=4,
        # patience=9999,
        # + utiliser: --layers 256 128 64 32
        # + mettre l2_lambda=0.0 ligne ~200
        verbose=True
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    train_loss, train_acc = network.evaluate(X_train, y_train_prepared)
    val_loss, val_acc = network.evaluate(X_val, y_val_prepared)

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

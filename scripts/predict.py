#!/usr/bin/env python
"""
Prediction script for the MLP classifier.

Usage:
    python predict.py data/data_test.csv    # Predict and evaluate on test set
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neural_network.network import Network


def detect_format(path):
    """Detect if CSV has header or not"""
    with open(path, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split(',')
        # If second column is M or B, it's evaluation.py format (no header)
        if len(parts) > 1 and parts[1] in ['M', 'B']:
            return 'no_header'
        return 'header'


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


def load_data_with_header(path):
    """Load data from CSV with header"""
    import pandas as pd

    df = pd.read_csv(path)

    if 'diagnosis' in df.columns:
        X = df.drop(['diagnosis'], axis=1).values
        y = df['diagnosis'].values
    else:
        X = df.values
        y = None

    return X, y


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <data.csv>")
        sys.exit(1)

    data_path = sys.argv[1]
    model_path = 'models/trained_model.pkl'
    norm_path = 'models/normalization_params.npy'

    # Check files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using train.py")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # Load model
    print(f"Loading model from {model_path}...")
    network = Network()
    network.load(model_path)

    # Load normalization params
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path, allow_pickle=True).item()
        mean = norm_data['mean']
        std = norm_data['std']
    else:
        print("Warning: Normalization params not found, using raw features")
        mean = 0
        std = 1

    # Detect format and load data
    print(f"Loading data from {data_path}...")
    data_format = detect_format(data_path)

    if data_format == 'no_header':
        X, y = load_data_no_header(data_path)
    else:
        X, y = load_data_with_header(data_path)

    # Normalize
    X_norm = (X - mean) / std

    # Make predictions
    print("Making predictions...")
    network.eval_mode()
    y_pred = network.forward(X_norm.T)

    # Get probabilities
    if y_pred.shape[0] == 2:
        y_pred_proba = y_pred[1, :]  # Probability of class 1 (Malignant)
        y_pred_classes = np.argmax(y_pred, axis=0)
    else:
        y_pred_proba = y_pred.flatten()
        y_pred_classes = (y_pred_proba > 0.5).astype(int)

    # Display results
    print("\n" + "=" * 60)
    print("Predictions")
    print("=" * 60)

    for i in range(min(10, len(y_pred_classes))):
        pred_label = "Malignant" if y_pred_classes[i] == 1 else "Benign"
        print(f"Sample {i+1}: {pred_label} (prob: {y_pred_proba[i]:.4f})")

    if len(y_pred_classes) > 10:
        print(f"... and {len(y_pred_classes) - 10} more samples")

    # Evaluate if labels available
    if y is not None:
        # Binary cross-entropy
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        bce_loss = -np.mean(
            y * np.log(y_pred_clipped) +
            (1 - y) * np.log(1 - y_pred_clipped)
        )

        accuracy = np.mean(y_pred_classes == y)

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"  Samples: {len(y)}")
        print(f"  Binary Cross-Entropy Loss: {bce_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({int(accuracy * len(y))}/{len(y)} correct)")

        # Scoring for evaluation.py
        print("\n" + "=" * 60)
        print("SCORING (evaluation.py)")
        print("=" * 60)
        if bce_loss > 0.35:
            score = 0
        elif bce_loss > 0.25:
            score = 1
        elif bce_loss > 0.18:
            score = 2
        elif bce_loss > 0.13:
            score = 3
        elif bce_loss > 0.08:
            score = 4
        else:
            score = 5

        print(f"  Loss: {bce_loss:.4f}")
        print(f"  Score: {score}/5")


if __name__ == "__main__":
    main()

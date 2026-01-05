"""Data preprocessing utilities for feature standardization and label encoding."""

import numpy as np


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

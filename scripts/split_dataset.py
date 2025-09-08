from operator import index

import pandas as pd
import numpy as np
import sys
import os

def load_dataset(filepath):
    """
    Load the breast cancer dataset from CSV file

    Args:
        filepath (str): Path to the CSV file

    Returns:
        tuple: (x, y) where x is features array and y is labels array
    """
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath, header=None)
        y = (df[1] == 'M').astype(int) # Malignant as 1, Benign as 0
        x = df.drop(1, axis=1)
        return x, y
    else:
        raise FileNotFoundError

def split_train_validation(x, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and validation sets

    Args:
        x (np.array): Features array
        y (np.array): Labels array
        test_size (float): Proportion for validation set
        random_state (int): Seed for reproducibility

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    np.random.seed(random_state)
    n_samples = len(x)
    indexs = np.random.permutation(n_samples)
    n_val = int(n_samples * test_size)
    val_indexs = indexs[:int(n_val)]
    train_indexs = indexs[int(n_val):]
    return x[train_indexs], y[train_indexs], x[val_indexs], y[val_indexs]

def save_splits(X_train, X_val, y_train, y_val, output_dir='data'):
    """
    Save train and validation splits to files

    Args:
        X_train, X_val: Features arrays
        y_train, y_val: Labels arrays
        output_dir (str): Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'train.npz'), X=X_train, y=y_train)
    np.savez(os.path.join(output_dir, 'val.npz'), X=X_val, y=y_val)
    pass

def main():
    """Main function to execute the split"""
    if len(sys.argv) != 2:
        print("Usage: python split_dataset.py <path_to_csv>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    x, y = load_dataset(dataset_path)
    X_train, X_val, y_train, y_val = split_train_validation(x.to_numpy(), y.to_numpy())
    save_splits(X_train, X_val, y_train, y_val)
    pass

if __name__ == "__main__":
    main()
"""Data loading utilities for CSV files with various formats."""

import numpy as np


def detect_csv_format(path):
    """
    Auto-detect CSV format by reading the first line.

    Priority: Checks column 1 for 'M'/'B' FIRST (no_header), then last column for 'diagnosis' (with_header).

    Returns:
        'with_header' if the file has a header row (last column is 'diagnosis')
        'no_header' if the file has no header (column 1 is 'M' or 'B')

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be read due to permissions
        ValueError: If the file is empty or format cannot be determined
    """
    try:
        with open(path, 'r') as f:
            first_line = f.readline().strip()

            if not first_line:
                raise ValueError(f"Empty CSV file: {path}")

            parts = first_line.split(',')

            if len(parts) < 2:
                raise ValueError(f"Invalid CSV format: expected at least 2 columns, got {len(parts)}")

            # Check column 1 for M/B FIRST (no_header format)
            if parts[1] in ['M', 'B']:
                return 'no_header'
            # Then check last column for 'diagnosis' (with_header format)
            elif parts[-1] == 'diagnosis':
                return 'with_header'
            else:
                raise ValueError(f"Cannot detect CSV format: column 1 is '{parts[1]}', last column is '{parts[-1]}'. Expected 'M'/'B' in column 1 or 'diagnosis' in last column.")

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {path}")
    except PermissionError:
        raise PermissionError(f"Permission denied when reading CSV file: {path}")


def load_train_val_data(train_path, val_path):
    """
    Load training and validation data from CSV files with headers.

    Args:
        train_path: Path to training data CSV with header
        val_path: Path to validation data CSV with header

    Returns:
        X_train, y_train, X_val, y_val: Training and validation features and labels

    Raises:
        ImportError: If pandas is not installed
        FileNotFoundError: If train_path or val_path does not exist
        ValueError: If diagnosis column is missing in either file
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for loading CSV files with headers. "
            "Install it with: pip install pandas"
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    if 'diagnosis' not in train_df.columns:
        raise ValueError(f"Missing 'diagnosis' column in training file: {train_path}")
    if 'diagnosis' not in val_df.columns:
        raise ValueError(f"Missing 'diagnosis' column in validation file: {val_path}")

    X_train = train_df.drop(['diagnosis'], axis=1).values
    y_train = train_df['diagnosis'].values
    X_val = val_df.drop(['diagnosis'], axis=1).values
    y_val = val_df['diagnosis'].values

    return X_train, y_train, X_val, y_val


def load_single_dataset(path):
    """
    Load a single dataset from CSV file with header.

    Args:
        path: Path to CSV file with header

    Returns:
        X, y: Features and labels (y is None if no diagnosis column)

    Raises:
        ImportError: If pandas is not installed
        FileNotFoundError: If path does not exist
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for loading CSV files with headers. "
            "Install it with: pip install pandas"
        )

    df = pd.read_csv(path)

    if 'diagnosis' in df.columns:
        X = df.drop(['diagnosis'], axis=1).values
        y = df['diagnosis'].values
    else:
        X = df.values
        y = None

    return X, y


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

import pandas as pd
import numpy as np  # Pour random seed
import sys
import os


def load_dataset(filepath):
    """
    Load the breast cancer dataset from CSV file
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    # Charger avec header (le dataset a normalement des noms de colonnes)
    df = pd.read_csv(filepath)

    # La colonne diagnosis contient M/B
    if 'diagnosis' in df.columns:
        y = (df['diagnosis'] == 'M').astype(int)
        x = df.drop(['diagnosis'], axis=1)
    else:
        # Si pas de header, assume colonne 1 comme dans votre version
        y = (df.iloc[:, 1] == 'M').astype(int)
        x = df.drop(df.columns[1], axis=1)

    # Supprimer la colonne ID s'il y en a une
    if 'id' in x.columns:
        x = x.drop(['id'], axis=1)
    elif x.columns[0].lower().startswith('id'):
        x = x.drop(x.columns[0], axis=1)

    return x, y


def split_train_validation(x, y, test_size=0.25, random_state=63465):
    """
    Split dataset into training and validation sets
    """
    np.random.seed(random_state)
    n_samples = len(x)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * test_size)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return (x.iloc[train_indices], x.iloc[val_indices],
            y.iloc[train_indices], y.iloc[val_indices])


def save_splits(X_train, X_val, y_train, y_val, output_dir='data'):
    """
    Save train and validation splits to CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Combiner features et labels pour sauvegarder en CSV
    train_df = X_train.copy()
    train_df['diagnosis'] = y_train

    val_df = X_val.copy()
    val_df['diagnosis'] = y_val

    # Sauvegarder en CSV
    train_path = os.path.join(output_dir, 'data_train.csv')
    val_path = os.path.join(output_dir, 'data_val.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Training set saved to: {train_path}")
    print(f"Validation set saved to: {val_path}")


def main():
    """Main function to execute the split"""
    if len(sys.argv) != 2:
        print("Usage: python split_data.py <path_to_csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    try:
        # Charger les données
        x, y = load_dataset(dataset_path)
        print(f"Dataset loaded: {x.shape[0]} samples, {x.shape[1]} features")

        # Diviser train/validation
        X_train, X_val, y_train, y_val = split_train_validation(x, y)

        # Réinitialiser les indices
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        # Sauvegarder (données brutes, normalisation dans train.py)
        save_splits(X_train, X_val, y_train, y_val)

        # Afficher les dimensions comme dans l'exemple du sujet
        print(f"x_train shape : ({X_train.shape[0]}, {X_train.shape[1]})")
        print(f"x_valid shape : ({X_val.shape[0]}, {X_val.shape[1]})")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

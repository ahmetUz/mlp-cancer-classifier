import pandas as pd
import numpy as np
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


def normalize_features(X_train, X_val):
    """
    Normalize features using training set statistics
    """
    # Calculer mean et std sur le training set seulement
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Éviter division par 0
    std = np.where(std == 0, 1, std)

    # Normaliser train et validation avec les mêmes paramètres
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    return X_train_norm, X_val_norm, mean, std


def split_train_validation(x, y, test_size=0.2, random_state=42):
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

        # Normaliser les features
        X_train_norm, X_val_norm, mean, std = normalize_features(
            X_train.values, X_val.values
        )

        # Reconvertir en DataFrame avec les noms de colonnes
        X_train = pd.DataFrame(X_train_norm, columns=x.columns)
        X_val = pd.DataFrame(X_val_norm, columns=x.columns)

        # Réinitialiser les indices
        X_train.reset_index(drop=True, inplace=True)
        X_val.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_val.reset_index(drop=True, inplace=True)

        # Sauvegarder
        save_splits(X_train, X_val, y_train, y_val)

        # Afficher les dimensions comme dans l'exemple du sujet
        print(f"x_train shape : ({X_train.shape[0]}, {X_train.shape[1]})")
        print(f"x_valid shape : ({X_val.shape[0]}, {X_val.shape[1]})")

        # Sauvegarder aussi les paramètres de normalisation
        np.savez(os.path.join('data', 'normalization_params.npz'),
                 mean=mean, std=std)
        print("Normalization parameters saved to: data/normalization_params.npz")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
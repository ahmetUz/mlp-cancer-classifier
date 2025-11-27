import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neural_network.network import Network


def load_data(train_path, val_path):
    """
    Load training and validation data from CSV files

    Args:
        train_path (str): Path to training CSV
        val_path (str): Path to validation CSV

    Returns:
        tuple: (X_train, y_train, X_val, y_val) as numpy arrays
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Separate features and labels
    X_train = train_df.drop(['diagnosis'], axis=1).values
    y_train = train_df['diagnosis'].values

    X_val = val_df.drop(['diagnosis'], axis=1).values
    y_val = val_df['diagnosis'].values

    # Convert labels to one-hot encoding for softmax (2 classes)
    y_train_onehot = np.zeros((len(y_train), 2))
    y_train_onehot[np.arange(len(y_train)), y_train.astype(int)] = 1

    y_val_onehot = np.zeros((len(y_val), 2))
    y_val_onehot[np.arange(len(y_val)), y_val.astype(int)] = 1

    return X_train, y_train_onehot, X_val, y_val_onehot


def main():
    """Train the neural network on breast cancer dataset"""

    # Paths to data
    train_path = 'data/data_train.csv'
    val_path = 'data/data_val.csv'

    # Check if data files exist
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Error: Training or validation data not found!")
        print("Please run split_dataset.py first to generate data files.")
        sys.exit(1)

    # Load data
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)
    print(f"x_train shape : ({X_train.shape[0]}, {X_train.shape[1]})")
    print(f"x_valid shape : ({X_val.shape[0]}, {X_val.shape[1]})")

    # Create network architecture
    network = Network()

    layer_configs = [
        {'input_size': X_train.shape[1], 'output_size': 24, 'activation': 'relu', 'weights_initializer': 'he'},
        {'input_size': 24, 'output_size': 24, 'activation': 'relu', 'weights_initializer': 'he'},
        {'input_size': 24, 'output_size': 2, 'activation': 'softmax', 'weights_initializer': 'xavier'}
    ]

    network.create_network(layer_configs)

    # Training parameters
    epochs = 300
    learning_rate = 0.1
    batch_size = 16
    patience = 30

    # Train the network
    network.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        verbose=True
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    train_loss, train_acc = network.evaluate(X_train, y_train)
    val_loss, val_acc = network.evaluate(X_val, y_val)

    print(f"\nFinal Results:")
    print(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save the trained model
    model_path = 'models/trained_model.pkl'
    network.save(model_path)

    print("\nModel saved successfully!")

    # Plot learning curves
    print("\nGenerating learning curves...")
    plot_learning_curves(network.history)


def plot_learning_curves(history):
    """
    Plot training and validation loss and accuracy curves

    Args:
        history (dict): Dictionary containing training history
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/learning_curves.png', dpi=100, bbox_inches='tight')
    print("Learning curves saved to: models/learning_curves.png")

    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()
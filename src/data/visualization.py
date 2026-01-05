"""Visualization utilities for training metrics."""

import os
import matplotlib.pyplot as plt


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

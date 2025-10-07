import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.neural_network.network import Network


def predict_from_csv(model_path, data_path, output_path=None):
    """
    Make predictions on a CSV file

    Args:
        model_path (str): Path to trained model
        data_path (str): Path to input CSV file
        output_path (str): Optional path to save predictions

    Returns:
        np.array: Predictions
    """
    # Load model
    print(f"Loading model from {model_path}...")
    network = Network()
    network.load(model_path)

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Check if diagnosis column exists (for evaluation)
    has_labels = 'diagnosis' in df.columns

    if has_labels:
        X = df.drop(['diagnosis'], axis=1).values
        y_true = df['diagnosis'].values
    else:
        X = df.values
        y_true = None

    # Make predictions
    print("Making predictions...")

    # Get raw predictions from network
    y_pred_raw = network.forward(X.T)

    # Check if softmax output (2 classes) or sigmoid (1 class)
    if y_pred_raw.shape[0] == 2:
        # Softmax: get class probabilities and predictions
        probabilities = y_pred_raw[1, :]  # Probability of malignant (class 1)
        predictions = np.argmax(y_pred_raw, axis=0)
    else:
        # Sigmoid: threshold at 0.5
        probabilities = y_pred_raw.flatten()
        predictions = (probabilities > 0.5).astype(int)

    # Display results
    print("\nPredictions:")
    print("-" * 60)
    for i in range(len(predictions)):
        pred_label = "Malignant" if predictions[i] == 1 else "Benign"
        print(f"Sample {i+1}: {pred_label} (probability: {probabilities[i]:.4f})")
        if has_labels:
            true_label = "Malignant" if y_true[i] == 1 else "Benign"
            correct = "" if predictions[i] == y_true[i] else ""
            print(f"           True: {true_label} {correct}")

    # Evaluate if labels are available
    if has_labels:
        loss, accuracy = network.evaluate(X, y_true)
        print("\n" + "=" * 60)
        print(f"Evaluation Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({int(accuracy * len(y_true))}/{len(y_true)} correct)")

    # Save predictions if output path is provided
    if output_path:
        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        })
        if has_labels:
            results_df['true_label'] = y_true
            results_df['correct'] = predictions == y_true

        results_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")

    return predictions, probabilities


def predict_single_sample(model_path, features):
    """
    Make prediction on a single sample

    Args:
        model_path (str): Path to trained model
        features (list or np.array): Feature values

    Returns:
        tuple: (prediction, probability)
    """
    # Load model
    network = Network()
    network.load(model_path)

    # Convert to numpy array
    X = np.array(features).reshape(1, -1)

    # Make prediction
    probability = network.predict_proba(X)[0]
    prediction = network.predict_classes(X, threshold=0.5)[0]

    return prediction, probability


def main():
    """Main function for making predictions"""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py <data.csv> [output.csv]")
        print("  python predict.py --model <model_path> --data <data.csv> [--output <output.csv>]")
        sys.exit(1)

    # Default model path
    model_path = 'models/trained_model.pkl'

    # Parse arguments
    if sys.argv[1] == '--model':
        if len(sys.argv) < 4:
            print("Error: --model requires a model path and --data argument")
            sys.exit(1)
        model_path = sys.argv[2]
        data_path = sys.argv[4] if sys.argv[3] == '--data' else sys.argv[3]
        output_path = sys.argv[6] if len(sys.argv) > 6 and sys.argv[5] == '--output' else None
    else:
        data_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train a model first using train.py")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # Make predictions
    predict_from_csv(model_path, data_path, output_path)


if __name__ == "__main__":
    main()
import argparse
import pickle
import numpy as np
import pandas as pd


def load_model(
    path: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["weights"], data["biases"], data["mean"], data["std"]


def forward(
    X: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]
) -> np.ndarray:
    a = X
    for i, (w, b) in enumerate(zip(weights, biases)):
        z = a @ w.T + b.T
        is_output = i == len(weights) - 1
        a = _softmax(z) if is_output else _relu(z)
    return a


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions with a trained MLP")
    parser.add_argument("model", type=str, help="Path to the saved model (.pkl)")
    parser.add_argument("dataset", type=str, help="Path to the dataset to predict on")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    weights, biases, mean, std = load_model(args.model)

    df = pd.read_csv(args.dataset, header=None)
    X = df.iloc[:, 2:].to_numpy(dtype=float)
    labels = df.iloc[:, 1].to_numpy()

    X = (X - mean) / std
    y = (labels == "M").astype(int)

    output = forward(X, weights, biases)
    predictions = output.argmax(axis=1)
    label_names = {0: "B", 1: "M"}

    correct = (predictions == y).sum()
    accuracy = correct / len(y) * 100
    loss = -np.mean(np.log(np.clip(output[np.arange(len(y)), y], 1e-15, 1.0)))

    print(f"Samples   : {len(y)}")
    print(f"Correct   : {correct}/{len(y)}")
    print(f"Accuracy  : {accuracy:.1f}%")
    print(f"Loss      : {loss:.4f}")
    print()
    print(f"{'#':<6} {'True':<6} {'Predicted':<10} {'P(B)':<8} {'P(M)':<8} {'OK'}")
    print("-" * 46)
    for i, (true, pred, probs) in enumerate(zip(y, predictions, output)):
        ok = "✓" if true == pred else "✗"
        print(
            f"{i:<6} {label_names[true]:<6} {label_names[pred]:<10} {probs[0]:.4f}   {probs[1]:.4f}   {ok}"
        )

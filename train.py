import argparse
import numpy as np
import pandas as pd

DEFAULT_LAYERS = [24, 24]


def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath, header=None)

    X = df.iloc[:, 2:].to_numpy(dtype=float)
    labels = df.iloc[:, 1].to_numpy()

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    y = (labels == "M").astype(int)

    return X, y


class Network:
    def __init__(self, layer_sizes: list[int]) -> None:
        self.layer_sizes = layer_sizes
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            # He initialization for ReLU activations
            w = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
            b = np.zeros((fan_out, 1))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.activations = [X]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ w.T + b.T
            is_output = i == len(self.weights) - 1
            a = self._softmax(z) if is_output else self._relu(z)
            self.activations.append(a)

        return self.activations[-1]

    def loss(self, output: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        y_truth = np.zeros_like(output)
        y_truth[np.arange(n), y] = 1.0
        output = np.clip(output, 1e-15, 1.0)
        return -np.sum(y_truth * np.log(output)) / n

    def backward(self, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n = len(y)
        output = self.activations[-1]

        y_true = np.zeros_like(output)
        y_true[np.arange(n), y] = 1.0

        # combined gradient of softmax + cross-entropy simplifies to this
        delta = output - y_true  # shape (n_rows, 2)

        dW: list[np.ndarray] = []
        db: list[np.ndarray] = []

        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]

            dW.insert(0, (delta.T @ a_prev) / n)
            db.insert(0, delta.sum(axis=0, keepdims=True).T / n)

            if i > 0:
                delta = delta @ self.weights[i]
                delta = delta * (self.activations[i] > 0)  # ReLU derivative

        return dW, db

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        # Subtract row max for numerical stability
        z = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> None:
        for epoch in range(1, epochs + 1):
            output = self.forward(X)
            dW, db = self.backward(y)

            for i in range(len(self.weights)):
                self.weights[i] -= lr * dW[i]
                self.biases[i] -= lr * db[i]

            if epoch % 100 == 0:
                current_loss = self.loss(output, y)
                predictions = output.argmax(axis=1)
                accuracy = (predictions == y).mean() * 100
                print(
                    f"epoch {epoch:>5}/{epochs}  loss: {current_loss:.4f}  accuracy: {accuracy:.1f}%"
                )

    def __repr__(self) -> str:
        lines = ["Network architecture:"]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            lines.append(
                f"  Layer {i + 1}: ({w.shape[1]} -> {w.shape[0]})  W{w.shape}  b{b.shape}"
            )
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multilayer perceptron")
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        metavar="N",
        help="Hidden layer sizes (default: 24 24)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_train.csv",
        metavar="FILE",
        help="Path to the training dataset (default: dataset_train.csv)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X, y = load_data(args.dataset)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes -> M: {y.sum()}, B: {(y == 0).sum()}")

    input_size = X.shape[1]
    output_size = 2
    layer_sizes = [input_size] + args.layer + [output_size]

    network = Network(layer_sizes)
    print(network)

    network.train(X, y, epochs=args.epochs, lr=args.learning_rate)

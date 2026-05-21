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
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            # He initialization for ReLU activations
            w = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
            b = np.zeros((fan_out, 1))
            self.weights.append(w)
            self.biases.append(b)

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

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_LAYERS = [24, 24]


def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath, header=None)

    X = df.iloc[:, 2:].to_numpy(dtype=float)
    labels = df.iloc[:, 1].to_numpy()

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    y = (labels == "M").astype(int)

    return X, y, mean, std


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

    @staticmethod
    def _compute_metrics(
        predictions: np.ndarray, y: np.ndarray
    ) -> tuple[float, float, float]:
        tp = ((predictions == 1) & (y == 1)).sum()
        fp = ((predictions == 1) & (y == 0)).sum()
        fn = ((predictions == 0) & (y == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return float(precision), float(recall), float(f1)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        best_val_loss = float("inf")
        best_weights = None
        best_biases = None
        patience_counter = 0
        n = len(y)
        effective_batch = n if batch_size is None else batch_size

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n)
            X_shuffled, y_shuffled = X[indices], y[indices]

            for start in range(0, n, effective_batch):
                X_batch = X_shuffled[start : start + effective_batch]
                y_batch = y_shuffled[start : start + effective_batch]
                self.forward(X_batch)
                dW, db = self.backward(y_batch)
                for i in range(len(self.weights)):
                    self.weights[i] -= lr * dW[i]
                    self.biases[i] -= lr * db[i]

            output = self.forward(X)
            train_loss = self.loss(output, y)
            train_preds = output.argmax(axis=1)
            train_acc = (train_preds == y).mean()
            train_precision, train_recall, train_f1 = self._compute_metrics(
                train_preds, y
            )
            history["train_loss"].append(train_loss)
            history["train_acc"].append(float(train_acc))
            history["train_precision"].append(train_precision)
            history["train_recall"].append(train_recall)
            history["train_f1"].append(train_f1)

            val_loss = float("nan")
            val_acc = float("nan")
            val_f1 = float("nan")
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_loss = self.loss(val_output, y_val)
                val_preds = val_output.argmax(axis=1)
                val_acc = (val_preds == y_val).mean()
                val_precision, val_recall, val_f1 = self._compute_metrics(
                    val_preds, y_val
                )
                history["val_loss"].append(val_loss)
                history["val_acc"].append(float(val_acc))
                history["val_precision"].append(val_precision)
                history["val_recall"].append(val_recall)
                history["val_f1"].append(val_f1)

                if patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = [w.copy() for w in self.weights]
                        best_biases = [b.copy() for b in self.biases]
                        patience_counter = 0
                    else:
                        patience_counter += 1

            msg = (
                f"epoch {epoch:02d}/{epochs}"
                f" - loss: {train_loss:.4f} - acc: {train_acc:.4f}"
                f" - precision: {train_precision:.4f} - recall: {train_recall:.4f} - f1: {train_f1:.4f}"
            )
            if X_val is not None:
                msg += (
                    f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                    f" - val_f1: {val_f1:.4f}"
                )
            print(msg)

            if patience is not None and patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)"
                )
                assert best_weights is not None and best_biases is not None
                self.weights = best_weights
                self.biases = best_biases
                break

        return history

    def save(self, path: str, mean: np.ndarray, std: np.ndarray) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "layer_sizes": self.layer_sizes,
                    "weights": self.weights,
                    "biases": self.biases,
                    "mean": mean,
                    "std": std,
                },
                f,
            )
        print(f"Model saved to {path}")

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
        help="Number of training epochs (default: 10000)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=60,
        metavar="N",
        help="Early stopping patience (epochs without val_loss improvement before stopping, default: 50)",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="dataset_validate.csv",
        metavar="FILE",
        help="Path to the validation dataset (default: dataset_validate.csv)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Mini-batch size (default: full dataset / batch gradient descent)",
    )
    return parser.parse_args()


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    print(f"\nConfusion matrix ({label}):")
    print(f"                Predicted B   Predicted M")
    print(f"  Actual B    |  {tn:^10d}  |  {fp:^10d}  |")
    print(f"  Actual M    |  {fn:^10d}  |  {tp:^10d}  |")


def plot_history(history: dict[str, list[float]], epochs: int) -> None:
    epoch_range = range(1, epochs + 1)
    has_val = bool(history["val_loss"])

    fig = plt.figure(figsize=(14, 10))
    ax_loss = fig.add_subplot(2, 2, 1)
    ax_acc = fig.add_subplot(2, 2, 2)
    ax_pr = fig.add_subplot(2, 2, 3)
    ax_f1 = fig.add_subplot(2, 2, 4)

    ax_loss.plot(epoch_range, history["train_loss"], label="Train")
    if has_val:
        ax_loss.plot(epoch_range, history["val_loss"], label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epoch_range, history["train_acc"], label="Train")
    if has_val:
        ax_acc.plot(epoch_range, history["val_acc"], label="Validation")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    ax_pr.plot(epoch_range, history["train_precision"], label="Train precision")
    ax_pr.plot(epoch_range, history["train_recall"], label="Train recall")
    if has_val:
        ax_pr.plot(
            epoch_range, history["val_precision"], label="Val precision", linestyle="--"
        )
        ax_pr.plot(
            epoch_range, history["val_recall"], label="Val recall", linestyle="--"
        )
    ax_pr.set_xlabel("Epoch")
    ax_pr.set_ylabel("Score")
    ax_pr.set_title("Precision & Recall")
    ax_pr.legend()

    ax_f1.plot(epoch_range, history["train_f1"], label="Train")
    if has_val:
        ax_f1.plot(epoch_range, history["val_f1"], label="Validation")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1 score")
    ax_f1.set_title("F1 Score")
    ax_f1.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    X, y, mean, std = load_data(args.dataset)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes -> M: {y.sum()}, B: {(y == 0).sum()}")

    X_val, y_val = None, None
    try:
        X_val, y_val, _, _ = load_data(args.val_dataset)
        # re-standardize using training mean/std
        X_val_raw = (
            pd.read_csv(args.val_dataset, header=None).iloc[:, 2:].to_numpy(dtype=float)
        )
        X_val = (X_val_raw - mean) / std
        print(f"Loaded {X_val.shape[0]} validation samples")
    except FileNotFoundError:
        print(
            f"Validation dataset '{args.val_dataset}' not found, skipping validation metrics."
        )

    input_size = X.shape[1]
    output_size = 2
    layer_sizes = [input_size] + args.layer + [output_size]

    network = Network(layer_sizes)
    print(network)

    history = network.train(
        X,
        y,
        epochs=args.epochs,
        lr=args.learning_rate,
        X_val=X_val,
        y_val=y_val,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    network.save("model.pkl", mean, std)

    train_preds = network.forward(X).argmax(axis=1)
    print_confusion_matrix(y, train_preds, "train")

    if X_val is not None and y_val is not None:
        val_preds = network.forward(X_val).argmax(axis=1)
        print_confusion_matrix(y_val, val_preds, "validation")

    plot_history(history, len(history["train_loss"]))

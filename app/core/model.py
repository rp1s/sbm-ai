from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

Array = NDArray[np.float64]


def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x: Array) -> Array:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_derivative(x: Array) -> Array:
    return 1.0 - np.tanh(x) ** 2


class SimilarityModel:
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        rng = np.random.default_rng(seed)
        combined_dim = input_dim * 3

        self.W1 = rng.normal(scale=0.1, size=(hidden_dim, combined_dim))
        self.b1 = np.zeros((hidden_dim, 1), dtype=np.float64)
        
        self.W2 = rng.normal(scale=0.1, size=(1, hidden_dim))
        self.b2 = np.zeros((1, 1), dtype=np.float64)

    def _prepare_input(self, humming: Array, song: Array) -> Array:
        diff = np.abs(humming - song)
        return np.concatenate([humming, song, diff], axis=1)

    def forward(self, humming: Array, song: Array) -> Tuple[Array, Array, Array]:
        x = self._prepare_input(humming, song).T
        z1 = self.W1.dot(x) + self.b1
        a1 = np.tanh(z1)
        z2 = self.W2.dot(a1) + self.b2
        y_pred = sigmoid(z2)
        return y_pred, a1, z1

    def predict(self, humming: Array, song: Array) -> Array:
        y_pred, _, _ = self.forward(humming, song)
        return y_pred.flatten()

    def _compute_loss(self, y_pred: Array, labels: Array, l2_lambda: float = 0.0) -> float:
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        cross_entropy = -np.mean(
            labels * np.log(y_pred) + (1.0 - labels) * np.log(1.0 - y_pred)
        )
        
        l2_loss = 0.5 * l2_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return cross_entropy + l2_loss

    def train_step(
        self,
        humming: Array,
        song: Array,
        labels: Array,
        lr: float = 0.01,
        l2_lambda: float = 0.0,
    ) -> float:
        labels = labels.reshape(1, -1)
        y_pred, a1, z1 = self.forward(humming, song)
        loss = self._compute_loss(y_pred, labels, l2_lambda=l2_lambda)

        m = humming.shape[0]
        
        dz2 = y_pred - labels
        dW2 = dz2.dot(a1.T) / m
        db2 = np.mean(dz2, axis=1, keepdims=True)

        da1 = self.W2.T.dot(dz2)
        dz1 = da1 * tanh_derivative(z1)
        dW1 = dz1.dot(self._prepare_input(humming, song)) / m
        db1 = np.mean(dz1, axis=1, keepdims=True)

        dW2 += l2_lambda * self.W2
        dW1 += l2_lambda * self.W1

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        return float(loss)

    def fit(
        self,
        dataset,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.01,
        l2_lambda: float = 0.0,
        validation_data=None,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        if early_stopping and validation_data is None:
            raise ValueError("Для ранней остановки нужен валидационный датасет")

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val_loss = np.inf
        best_weights = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for humming_batch, song_batch, labels_batch in dataset.batch_iterator(batch_size=batch_size, seed=epoch):
                loss = self.train_step(humming_batch, song_batch, labels_batch, lr=lr, l2_lambda=l2_lambda)
                epoch_losses.append(loss)

            train_loss = float(np.mean(epoch_losses))
            train_pred = self.predict(dataset.humming_vectors, dataset.song_vectors)
            train_acc = float(np.mean((train_pred >= 0.5).astype(np.float64) == dataset.labels))

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if validation_data is not None:
                val_pred = self.predict(validation_data.humming_vectors, validation_data.song_vectors)
                val_loss = self._compute_loss(val_pred.reshape(1, -1), validation_data.labels, l2_lambda=l2_lambda)
                val_acc = float(np.mean((val_pred >= 0.5).astype(np.float64) == validation_data.labels))
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if restore_best_weights:
                        best_weights = {
                            "W1": self.W1.copy(),
                            "b1": self.b1.copy(),
                            "W2": self.W2.copy(),
                            "b2": self.b2.copy(),
                        }
                else:
                    patience_counter += 1

                if early_stopping and patience_counter >= patience:
                    if verbose:
                        print(f"Ранняя остановка на эпохе {epoch}: ошибка валидации не улучшалась {patience} эпох")
                    break
            else:
                history["val_loss"].append(float(np.nan))
                history["val_acc"].append(float(np.nan))

            if verbose and epoch % 10 == 0:
                msg = f"Эпоха {epoch}/{epochs}: ошибка={train_loss:.4f}, точность={train_acc:.4f}"
                if validation_data is not None:
                    msg += f", ошибка_валидации={val_loss:.4f}, точность_валидации={val_acc:.4f}"
                print(msg)

        if restore_best_weights and best_weights is not None:
            self.W1 = best_weights["W1"]
            self.b1 = best_weights["b1"]
            self.W2 = best_weights["W2"]
            self.b2 = best_weights["b2"]

        return history

    def save(self, path: str) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str, input_dim: int, hidden_dim: int) -> SimilarityModel:
        data = np.load(path)
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim)
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        return model

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class PairDataset:
    def __init__(self, humming_vectors: Array, song_vectors: Array, labels: Array):
        assert humming_vectors.shape[0] == song_vectors.shape[0] == len(labels)
        
        self.humming_vectors = humming_vectors
        self.song_vectors = song_vectors
        self.labels = labels
        self.n_samples = len(labels)

    @classmethod
    def synthetic(
        cls,
        n_pairs: int = 2000,
        input_dim: int = 32,
        match_ratio: float = 0.5,
        noise_scale: float = 0.12,
        seed: int = 42,
    ) -> PairDataset:
        rng = np.random.default_rng(seed)
        n_positive = int(n_pairs * match_ratio)
        n_negative = n_pairs - n_positive

        humming_positive = rng.normal(size=(n_positive, input_dim))
        song_positive = humming_positive + rng.normal(scale=noise_scale, size=(n_positive, input_dim))
        labels_positive = np.ones(n_positive, dtype=np.float64)

        humming_negative = rng.normal(size=(n_negative, input_dim))
        song_negative = rng.normal(size=(n_negative, input_dim))
        labels_negative = np.zeros(n_negative, dtype=np.float64)

        humming = np.concatenate([humming_positive, humming_negative], axis=0)
        songs = np.concatenate([song_positive, song_negative], axis=0)
        labels = np.concatenate([labels_positive, labels_negative], axis=0)

        permutation = rng.permutation(n_pairs)
        humming = humming[permutation]
        songs = songs[permutation]
        labels = labels[permutation]

        return cls(humming, songs, labels)

    def split(self, split_ratio: float = 0.8, seed: int = 42) -> Tuple[PairDataset, PairDataset]:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.n_samples)
        split_idx = int(self.n_samples * split_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        train_set = PairDataset(
            self.humming_vectors[train_idx],
            self.song_vectors[train_idx],
            self.labels[train_idx],
        )
        val_set = PairDataset(
            self.humming_vectors[val_idx],
            self.song_vectors[val_idx],
            self.labels[val_idx],
        )

        return train_set, val_set

    def batch_iterator(
        self, batch_size: int = 64, seed: int = 42
    ) -> Iterator[Tuple[Array, Array, Array]]:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.n_samples)
        
        for start in range(0, self.n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield (
                self.humming_vectors[batch_idx],
                self.song_vectors[batch_idx],
                self.labels[batch_idx],
            )

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return f"PairDataset(n_samples={self.n_samples}, input_dim={self.humming_vectors.shape[1]})"

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.core import PairDataset, extract_audio_features
from app.settings import INPUT_DIM, CACHE_DIR

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset_from_json(self, json_path: Path) -> Optional[PairDataset]:
        try:
            json_path = json_path.resolve()
            with open(json_path, "r", encoding="utf-8") as f:
                pairs = json.load(f)

            if not isinstance(pairs, list):
                pairs = [pairs]

            return self._load_pairs(pairs, base_dir=json_path.parent)
        except Exception as e:
            logger.error(f"Не удалось загрузить JSON датасета {json_path}: {e}")
            return None

    def _load_pairs(self, pairs: list[dict], base_dir: Path) -> Optional[PairDataset]:
        humming_vectors = []
        song_vectors = []
        labels = []

        for pair in pairs:
            humming_path = self._resolve_audio_path(base_dir, pair["humming_path"])
            song_path = self._resolve_audio_path(base_dir, pair["song_path"])
            label = float(pair["label"])

            if not humming_path.exists() or not song_path.exists():
                logger.warning(f"Аудиофайлы не найдены: {humming_path}, {song_path}")
                continue

            try:
                humming_feat = self.extract_features(humming_path)
                song_feat = self.extract_features(song_path)

                humming_vectors.append(humming_feat)
                song_vectors.append(song_feat)
                labels.append(label)
            except Exception as e:
                logger.error(f"Ошибка извлечения признаков из {humming_path}: {e}")
                continue

        if not labels:
            logger.warning(f"Не загружено ни одной корректной пары из {base_dir}")
            return None

        dataset = PairDataset(
            np.array(humming_vectors, dtype=np.float64),
            np.array(song_vectors, dtype=np.float64),
            np.array(labels, dtype=np.float64),
        )
        logger.info(f"Загружен датасет: {dataset.n_samples} пар из {base_dir}")
        return dataset

    def extract_features(self, audio_path: Path) -> np.ndarray:
        audio_path = audio_path.resolve()
        cache_path = self._cache_path(audio_path)

        if self.use_cache and cache_path.exists():
            try:
                data = np.load(cache_path)
                return data["features"].astype(np.float64)
            except Exception as e:
                logger.warning(f"Не удалось прочитать кеш {cache_path}: {e}")

        features = extract_audio_features(audio_path, INPUT_DIM)

        if self.use_cache:
            np.savez(cache_path, features=features)

        return features

    def _cache_path(self, audio_path: Path) -> Path:
        stat = audio_path.stat()
        cache_key = "|".join(
            [
                str(audio_path),
                str(stat.st_size),
                str(int(stat.st_mtime_ns)),
                str(INPUT_DIM),
            ]
        )
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{audio_path.stem}_{digest}.npz"

    @staticmethod
    def _resolve_audio_path(base_dir: Path, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return base_dir / path

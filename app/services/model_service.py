from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from app.core import PairDataset, SimilarityModel, extract_audio_features
from app.settings import (
    INPUT_DIM,
    HIDDEN_DIM,
    BATCH_SIZE,
    LEARNING_RATE,
    L2_LAMBDA,
    PATIENCE,
    MODEL_DIR,
    HISTORY_DIR,
)

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        self.model: Optional[SimilarityModel] = None

    def create_new_model(self) -> SimilarityModel:
        self.model = SimilarityModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        logger.info(f"Создана новая модель: {INPUT_DIM} -> {HIDDEN_DIM} -> 1")
        return self.model

    def train(
        self,
        train_dataset: PairDataset,
        val_dataset: Optional[PairDataset] = None,
        epochs: int = 500,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        dataset_name: str = "",
        verbose: bool = True,
    ) -> dict:
        if self.model is None:
            self.create_new_model()

        logger.info(f"Начало обучения: {train_dataset.n_samples} примеров")

        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            l2_lambda=L2_LAMBDA,
            validation_data=val_dataset,
            early_stopping=True,
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=verbose,
        )

        if val_dataset:
            val_pred = self.model.predict(
                val_dataset.humming_vectors,
                val_dataset.song_vectors,
            )
            accuracy = float(
                np.mean((val_pred >= 0.5).astype(np.float64) == val_dataset.labels)
            )
            val_loss = float(history["val_loss"][-1]) if history["val_loss"] else 0.0
        else:
            accuracy = float(history["train_acc"][-1]) if history["train_acc"] else 0.0
            val_loss = 0.0

        epochs_trained = len(history["train_loss"])

        logger.info(f"Обучение завершено: точность={accuracy:.4f}, эпох={epochs_trained}")

        saved = self._save_model_files(
            dataset_name=dataset_name,
            accuracy=accuracy,
            val_loss=val_loss,
            epochs=epochs_trained,
            batch_size=batch_size,
            learning_rate=lr,
        )

        return {
            "accuracy": accuracy,
            "val_loss": val_loss,
            "epochs": epochs_trained,
            "model_path": str(saved["model_path"]),
            "history_model_path": str(saved["history_model_path"]),
            "metadata_path": str(saved["metadata_path"]),
            "history": history,
        }

    def load_model(self, model_path: str) -> bool:
        try:
            self.model = SimilarityModel.load(
                model_path,
                input_dim=INPUT_DIM,
                hidden_dim=HIDDEN_DIM,
            )
            logger.info(f"Модель загружена из {model_path}")
            return True
        except Exception as e:
            logger.error(f"Не удалось загрузить модель: {e}")
            return False

    def predict(self, humming_path: Path, song_path: Path) -> float:
        if self.model is None:
            raise ValueError("Модель не загружена")

        humming_feat = extract_audio_features(humming_path, target_dim=INPUT_DIM)
        song_feat = extract_audio_features(song_path, target_dim=INPUT_DIM)

        score = self.model.predict(
            humming_feat.reshape(1, -1),
            song_feat.reshape(1, -1),
        )[0]

        return float(score)

    def get_best_model_path(self) -> Optional[Path]:
        out_model = MODEL_DIR / "sbm.model.npz"
        if out_model.exists():
            return out_model

        history_models = sorted(HISTORY_DIR.glob("model_*.npz"), reverse=True)
        if history_models:
            return history_models[0]

        return None

    def _save_model_files(
        self,
        dataset_name: str,
        accuracy: float,
        val_loss: float,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> dict[str, Path]:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        out_path = MODEL_DIR / "sbm.model.npz"
        self.model.save(str(out_path))

        history_path = HISTORY_DIR / f"model_{timestamp}.npz"
        self.model.save(str(history_path))

        metadata_path = HISTORY_DIR / f"model_{timestamp}.json"
        metadata = {
            "название_датасета": dataset_name,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "эпохи": epochs,
            "размер_батча": batch_size,
            "скорость_обучения": learning_rate,
            "путь_модели": str(out_path),
            "путь_истории": str(history_path),
            "создано": datetime.now().isoformat(),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Модель сохранена в {out_path}")
        return {
            "model_path": out_path,
            "history_model_path": history_path,
            "metadata_path": metadata_path,
        }


def evaluate(model: SimilarityModel, dataset: PairDataset) -> dict:
    predictions = model.predict(dataset.humming_vectors, dataset.song_vectors)
    labels = dataset.labels
    predicted_classes = (predictions >= 0.5).astype(np.float64)
    accuracy = np.mean(predicted_classes == labels)
    average_score = float(np.mean(predictions))

    return {
        "accuracy": float(accuracy),
        "average_score": average_score,
    }

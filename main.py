from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from app.services import DatasetService, ModelService
from app.settings import (
    BATCH_SIZE,
    EPOCHS,
    INPUT_DIM,
    LEARNING_RATE,
    MODEL_DIR,
    VAL_SPLIT,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SBM AI: обучение и проверка модели сопоставления напева с песней.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="Обучить модель по JSON-разметке.")
    train.add_argument("json", type=Path, help="Путь к JSON-файлу с парами audio/label.")
    train.add_argument("--val-json", type=Path, help="Отдельный JSON для валидации.")
    train.add_argument("--split", type=float, default=1.0 - VAL_SPLIT, help="Доля train при auto split.")
    train.add_argument("--epochs", type=int, default=EPOCHS, help="Количество эпох.")
    train.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Размер батча.")
    train.add_argument("--lr", type=float, default=LEARNING_RATE, help="Скорость обучения.")
    train.add_argument("--no-cache", action="store_true", help="Отключить кеш признаков аудио.")
    train.add_argument("--quiet", action="store_true", help="Не печатать прогресс обучения.")

    predict = subparsers.add_parser("predict", help="Проверить пару аудиофайлов.")
    predict.add_argument("--model", type=Path, default=MODEL_DIR / "sbm.model.npz", help="Путь к .npz модели.")
    predict.add_argument("--humming", type=Path, required=True, help="Путь к аудио с напевом.")
    predict.add_argument("--song", type=Path, required=True, help="Путь к аудио песни.")
    predict.add_argument("--no-cache", action="store_true", help="Отключить кеш признаков аудио.")

    subparsers.add_parser("info", help="Показать текущую модель, если она есть.")
    return parser


def train_command(args: argparse.Namespace) -> int:
    dataset_service = DatasetService(use_cache=not args.no_cache)
    model_service = ModelService()

    dataset = dataset_service.load_dataset_from_json(args.json)
    if dataset is None:
        print(f"Не удалось загрузить датасет: {args.json}", file=sys.stderr)
        return 1

    if args.val_json:
        val_data = dataset_service.load_dataset_from_json(args.val_json)
        if val_data is None:
            print(f"Не удалось загрузить валидационный датасет: {args.val_json}", file=sys.stderr)
            return 1
        train_data = dataset
    else:
        train_data, val_data = dataset.split(args.split)

    if train_data.n_samples == 0 or val_data.n_samples == 0:
        print("После разделения датасета train или val оказался пустым.", file=sys.stderr)
        return 1

    print(f"Датасет: {args.json}")
    print(f"Train: {train_data.n_samples} пар, val: {val_data.n_samples} пар")
    print(f"Эпохи: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}")

    model_service.create_new_model()
    result = model_service.train(
        train_data,
        val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_name=args.json.stem,
        verbose=not args.quiet,
    )

    print("\nОбучение завершено")
    print(f"Точность: {result['accuracy']:.4f}")
    print(f"Ошибка валидации: {result['val_loss']:.6f}")
    print(f"Эпох обучено: {result['epochs']}")
    print(f"Модель: {result['model_path']}")
    print(f"История: {result['history_model_path']}")
    print(f"Метаданные: {result['metadata_path']}")
    return 0


def predict_command(args: argparse.Namespace) -> int:
    if not args.model.exists():
        print(f"Модель не найдена: {args.model}", file=sys.stderr)
        return 1

    dataset_service = DatasetService(use_cache=not args.no_cache)
    model_service = ModelService()

    if not model_service.load_model(str(args.model)):
        print(f"Не удалось загрузить модель: {args.model}", file=sys.stderr)
        return 1

    humming = dataset_service.extract_features(args.humming).reshape(1, INPUT_DIM)
    song = dataset_service.extract_features(args.song).reshape(1, INPUT_DIM)
    score = float(model_service.model.predict(humming, song)[0])
    confidence = score if score >= 0.5 else 1.0 - score
    verdict = "совпадает" if score >= 0.5 else "не совпадает"

    print(f"Оценка сходства: {score:.4f}")
    print(f"Вердикт: {verdict}")
    print(f"Уверенность: {confidence:.2%}")
    return 0


def info_command() -> int:
    model_service = ModelService()
    model_path = model_service.get_best_model_path()
    if model_path is None:
        print("Модель не найдена. Сначала запустите обучение.")
        return 1

    print(f"Текущая модель: {model_path}")
    data = np.load(model_path)
    print(f"W1: {data['W1'].shape}, b1: {data['b1'].shape}")
    print(f"W2: {data['W2'].shape}, b2: {data['b2'].shape}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        return train_command(args)
    if args.command == "predict":
        return predict_command(args)
    if args.command == "info":
        return info_command()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

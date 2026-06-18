# SBM Model

Данный проект был создан за один вечер мной и Клодом для проверки работоспособности идеи. 
Главная ценность проекта это датасет, который создаётся вручную.

## Возможности

- обработка WAV, MP3, OGG и FLAC
- обучение модели по JSON-файлу с парами аудио
- кеширование аудиопризнаков в `data/cache/`
- сохранение текущей модели в `out/sbm.model.npz`
- сохранение копий моделей и JSON-метаданных в `history/`
- проверка пары `напев + песня` из командной строки

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Для MP3/OGG/FLAC нужен FFmpeg:

```bash
brew install ffmpeg
```

## Формат JSON

JSON должен быть списком пар:

```json
[
  {
    "id": "example_1",
    "humming_path": "audio/humming/example.wav",
    "song_path": "audio/song/song.wav",
    "label": 1
  }
]
```

Пути могут быть абсолютными или относительными от папки, где лежит JSON.
`label = 1` означает совпадение, `label = 0` означает несовпадение.

## Обучение

```bash
python3 main.py train data/chant/generated/train.json
```

С отдельной валидацией:

```bash
python3 main.py train data/chant/generated/train.json --val-json data/chant/generated/val.json
```

Параметры:

```bash
python3 main.py train data/chant/generated/train.json \
  --epochs 500 \
  --batch-size 16 \
  --lr 0.02
```

Отключить кеш:

```bash
python3 main.py train data/chant/generated/train.json --no-cache
```

После обучения создаются:

- `out/sbm.model.npz` - текущая модель
- `history/model_YYYY-MM-DD_HH-MM-SS.npz` - копия модели
- `history/model_YYYY-MM-DD_HH-MM-SS.json` - метаданные обучения

## Проверка пары

```bash
python3 main.py predict \
  --humming path/to/humming.wav \
  --song path/to/song.mp3
```

С явным путём к модели:

```bash
python3 main.py predict \
  --model history/model_2026-06-18_10-00-00.npz \
  --humming path/to/humming.wav \
  --song path/to/song.mp3
```

## Информация о модели

```bash
python3 main.py info
```

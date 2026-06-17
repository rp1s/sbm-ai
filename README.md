# sbm-ai

## Формат данных JSON для `data/chant`

Лучше всего использовать WAV, MP3 или OGG-файлы, а не заранее вычисленные векторы. Код автоматически извлекает признаки из аудио и кеширует их в `data/cache/`.

Сейчас скрипт `src/train.py` ожидает отдельные папки:

- `data/chant/train/` — набор для обучения
- `data/chant/val/` — набор для проверки

Пример формата с путями:

```json
{
  "id": "example_wav_1",
  "humming_path": "audio/hum_example.wav",
  "song_path": "audio/song_example.wav",
  "label": 1
}
```

Файл может содержать один объект или массив объектов:

```json
[
  {
    "id": "example_wav_1",
    "humming_path": "audio/hum_example.wav",
    "song_path": "audio/song_example.wav",
    "label": 1
  },
  {
    "id": "example_wav_2",
    "humming_path": "audio/hum2.wav",
    "song_path": "audio/song2.wav",
    "label": 0
  }
]
```

Поля:

- `id` — уникальный идентификатор примера
- `humming_path` — относительный путь к WAV-файлу напева внутри `data/chant`
- `song_path` — относительный путь к WAV-файлу песни внутри `data/chant`
- `label` — бинарная метка: `1` означает, что песня подходит к напеву, `0` означает, что не подходит

Если у вас уже есть числовые векторы, это тоже поддерживается, но предпочтительнее сразу хранить WAV и кешировать признаки.

## Запуск обучения

```bash
cd /Users/petrov/Documents/sbm-ai
python3 src/train.py
```
В `src/train.py` реализованы:
- обучение на нескольких эпохах
- отчёт по train/validation loss и accuracy каждую эпоху
- L2-регуляризация для защиты от переобучения
- ранняя остановка (early stopping) по валидационной потере
После обучения модель сохраняется в `src/model_params.npz`.

Если нужно, можно экспортировать веса в JSON для Go-интерфейса командой:

```bash
cd /Users/petrov/Documents/sbm-ai
python3 src/export_weights.py
```

Это создаст файл `src/model_weights.json` только при необходимости.

## Go-интерфейс модели

Если хотите выполнить предсказание на Go, используйте `go_model.go`.

Запуск:

```bash
go run go_model.go src/model_weights.json data/chant/example.json output.json
```

В результате `output.json` будет содержать процент совпадения.

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
MODEL_DIR = ROOT_DIR / "out"
HISTORY_DIR = ROOT_DIR / "history"

INPUT_DIM = 32
HIDDEN_DIM = 64

BATCH_SIZE = 16
LEARNING_RATE = 0.02
L2_LAMBDA = 1e-4
EPOCHS = 500
PATIENCE = 30
VAL_SPLIT = 0.2

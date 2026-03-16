import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Data file paths — set dynamically via set_data_paths()
CURRENT_DATA_DIR = None
TRAIN_SKUS_FILE = None
VAL_SKUS_FILE = None
TEST_SKUS_FILE = None


def set_data_paths(data_dir=None):
    """Update global data paths. Call before modules that depend on these paths."""
    global TRAIN_SKUS_FILE, VAL_SKUS_FILE, TEST_SKUS_FILE, CURRENT_DATA_DIR
    if data_dir is not None:
        d = Path(data_dir).resolve()
        CURRENT_DATA_DIR = d
        TRAIN_SKUS_FILE = d / "train.json"
        VAL_SKUS_FILE = d / "val.json"
        TEST_SKUS_FILE = d / "test.json"

def get_data_file_path(filename):
    """Get path to a data file in the current data directory."""
    if CURRENT_DATA_DIR is not None:
        return CURRENT_DATA_DIR / filename
    return PROJECT_ROOT / "synthetic" / filename

def load_json(file_path, fallback_name):
    """Load JSON from file_path, falling back to data dir."""
    if file_path is not None and Path(file_path).exists():
        return json.load(open(file_path))
    return json.load(open(get_data_file_path(fallback_name)))

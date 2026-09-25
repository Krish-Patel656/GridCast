"""Configuration for the F1 finishing-order ranking pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"

FIRST_SEASON = 2014
COLLECT_SEASONS = list(range(FIRST_SEASON, 2027))

TRAINING_CSV = PROCESSED_DIR / "training_data.csv"
DATASET_META_JSON = PROCESSED_DIR / "dataset_meta.json"
ENCODERS_JSON = PROCESSED_DIR / "encoders.json"
SCALER_PATH = PROCESSED_DIR / "feature_scaler.joblib"
METADATA_JSON = MODELS_DIR / "metadata.json"
MODEL_PATH = MODELS_DIR / "race_predictor.pt"

# Every feature below is knowable before the race starts.
NUMERIC_FEATURES = [
    "grid_position",
    "has_grid",
    "pace_prior",
    "grid_percentile",
    "grid_vs_teammate",
    "driver_points_share",
    "driver_position_before",
    "constructor_points_share",
    "constructor_position_before",
    "avg_finish_last_3",
    "avg_finish_last_5",
    "team_avg_finish_last_3",
    "circuit_avg_finish",
    "team_circuit_avg_finish",
    "dnf_rate_last_5",
    "team_dnf_rate_last_5",
    "season_progress",
]

CATEGORICAL_FEATURES = ["driver_code", "team_id", "circuit_id"]
TARGET_COLUMN = "finish_position"

EMBEDDING_DIM = 8
HIDDEN_DIMS = [64, 32]
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 120
EARLY_STOPPING_PATIENCE = 25
VAL_FRACTION = 0.15
RANDOM_SEED = 42

# How far the model may move a driver from their starting slot. Real recoveries
# and collapses of more than this are luck, not something to fit.
MAX_POSITION_SHIFT = 8.0

# Fraction of augmented training copies where the grid is hidden, so the model
# stays calibrated for races predicted before qualifying has run.
GRID_MASK_FRACTION = 0.5

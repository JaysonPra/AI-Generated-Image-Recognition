from pathlib import Path

DATASET_NAME = "hassnainzaidi/ai-art-vs-human-art"

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_SAVE_DIR = ROOT_DIR / "models"
EXPERIMENTATION_DIR = ROOT_DIR / "config" / "experimentation"

MODEL_NAME = "AI-Image-Classifier"
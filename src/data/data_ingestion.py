from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import os
from config.config import RAW_DATA_DIR, DATASET_NAME
import zipfile

load_dotenv()

def run_ingestion():
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(RAW_DATA_DIR):
        os.mkdir(RAW_DATA_DIR)
        print("Created data directory for ingestions...")

    api.dataset_download_files(DATASET_NAME, path=RAW_DATA_DIR, quiet=False)

def extract(zip_file):
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(path=RAW_DATA_DIR)
        
        os.remove(zip_file)
        print("Zip file extracted successfully...")
    else:
        print("Zip File not found...")

if __name__ == "__main__":
    run_ingestion()
    extract(RAW_DATA_DIR / "ai-art-vs-human-art.zip")
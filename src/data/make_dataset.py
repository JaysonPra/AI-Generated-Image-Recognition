from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
import pandas as pd
from sklearn.model_selection import train_test_split

def make_dataset(accepted_extensions, data_directory):
    data = []

    for folder in ['AiArtData', 'RealArt']:
        path_to_folder = data_directory / folder

        label = 1 if folder == 'AiArtData' else 0

        for file_path in path_to_folder.iterdir():
            if file_path.suffix.lower() in accepted_extensions:
                data.append({
                    'filepath': str(file_path.absolute()),
                    'label': label
                })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = make_dataset(
        accepted_extensions = ('.png', '.jpg', '.jpeg', '.webp'),
        data_directory = RAW_DATA_DIR
    )

    if not df.empty:
        train_val_df, test_df = train_test_split(
            df, test_size=0.15, stratify=df["label"], random_state=42
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=0.176, stratify=train_val_df["label"], random_state=42
        )

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
import pandas as pd
from sklearn.model_selection import train_test_split

def make_dataset(accepted_extensions, data_directory):
    """Function to make a DataFrame for image paths and labels

    Args:
        accepted_extensions (tuple): A tuple of extensions to be accepted. Example: ('.png', '.jpeg')
        data_directory (str): Directory of the Images

    Returns:
        pd.DataFrame: DataFrame with image path and label
    """
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
        train_df, test_df = train_test_split(
            df, test_size=0.20, stratify=df["label"], random_state=42
        )

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

        print("Files written succesfully...")

import torch
import pandas as pd
from src.data.dataloader import get_dataloader
from src.data.dataset import ImageDataset
from src.data.transforms import get_transformation
from src.model.resnet import get_trainable_model
from src.utils.common_utils import run_epoch
from config.config import PROCESSED_DATA_DIR

def test_model(config, model_path):
    """Tests the given model using the test.csv file

    Args:
        config (dict): The loaded YAML config file
        model_path (str): Exact path of the .pth file for testing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = PROCESSED_DATA_DIR / "test.csv"
    test_df = pd.read_csv(test_csv)
    test_dataset = ImageDataset(test_df, transform=get_transformation(config, is_training=False))
    test_loader = get_dataloader(
        dataset=test_dataset,
        batch_size=config["trainig"]["batch_size"],
        shuffle=False
    )

    model = get_trainable_model(config).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_acc = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device
    )

    print("-- Test Results --")
    print(f"Test loss: {test_loss}")    
    print(f"Test Accuracy: {test_acc}")
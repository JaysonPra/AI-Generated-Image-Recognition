import torch
from torch import nn
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

from src.data.dataloader import get_dataloader
from src.data.dataset import ImageDataset
from config.config import PROCESSED_DATA_DIR, MODEL_SAVE_DIR
from src.data.transforms import get_transformation
from src.model.resnet import get_trainable_model
from src.utils.train_utils import get_optimizer
from src.utils.common_utils import run_epoch

def _run_training_session(config, train_loader, val_loader, save_path):
    """Trains a single model

    Args:
        config (dict): The loaded YAML config file
        train_loader (torch.utils.data.DataLoader): DataLoader for the Training Dataset
        val_loader (torch.utils.data.DataLoader): DataLoader for the Validation Dataset
        save_path (str): Path to save model

    Returns:
        tuple: Best Accuracy, Model Object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_trainable_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)
    best_acc = 0.0

    for epoch in range(1, config["training"]["epochs"]+1):
        run_epoch(model, train_loader, criterion, device, optimizer)
        _, val_acc = run_epoch(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict()}, save_path)
 
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return best_acc, model

def train_model_cv(config):
    """Model Training with Cross Validation

    Args:
        config (dict): The loaded YAML config file

    Returns:
        tuple: Mean Accuracy, Standard Deviation of Accuracy
    """
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    df = pd.read_csv(train_csv)
    labels = df['label'].values

    experiment_name = config["experiment"].get("experiment_name", "default_run")
    save_dir = MODEL_SAVE_DIR / "checkpoints" / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset_full = ImageDataset(df, transform=get_transformation(config, is_training=True))
    val_dataset_full = ImageDataset(df, transform=get_transformation(config, is_training=False))

    skf = StratifiedKFold(n_splits=config["training"]["n_splits"])
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_save_path = save_dir / f"resnet_fold_{fold}_best.pth"
        
        train_loader = get_dataloader(
            Subset(train_dataset_full, train_idx), 
            batch_size=config["training"]["batch_size"],
            shuffle=True
        )

        val_loader = get_dataloader(
            Subset(val_dataset_full, val_idx), 
            batch_size=config["training"]["batch_size"], 
            shuffle=False
        )

        acc, _ = _run_training_session(config, train_loader, val_loader, fold_save_path)
        fold_results.append(acc)
        torch.cuda.empty_cache()

    return np.mean(fold_results), np.std(fold_results)

def train_model_final(config):
    """Train model once and evaluate on the Test file

    Args:
        config (dict): The loaded YAML config file

    Returns:
        tuple: Accuracy of Model, Model Object
    """
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    test_csv = PROCESSED_DATA_DIR / "test.csv"
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    experiment_name = config["experiment"].get("experiment_name", "default_run")
    save_path = MODEL_SAVE_DIR / "final" / f"{experiment_name}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_dataset_full = ImageDataset(train_df, transform=get_transformation(config, is_training=True))
    test_dataset_full = ImageDataset(test_df, transform=get_transformation(config, is_training=False))
    
    train_loader = get_dataloader(train_dataset_full, batch_size=config["training"]["batch_size"], shuffle=True)
    test_loader = get_dataloader(test_dataset_full, batch_size=config["training"]["batch_size"], shuffle=False)

    best_acc, model = _run_training_session(config, train_loader, test_loader, save_path)
    return best_acc, model
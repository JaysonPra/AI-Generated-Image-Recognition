import torch
from torch import nn
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

from src.data.dataloader import get_dataloader
from src.data.dataset import ImageDataset
from config.config import PROCESSED_DATA_DIR
from src.data.transforms import get_transformation
from src.model.resnet import get_trainable_model
from src.utils.train_utils import get_optimizer
from src.utils.common_utils import run_epoch

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    train_csv = PROCESSED_DATA_DIR / "train.csv"
    df = pd.read_csv(train_csv)
    labels = df['label'].values

    skf = StratifiedKFold(n_splits=config["training"]["n_splits"])

    train_dataset_full = ImageDataset(df, transform=get_transformation(config, is_training=True))
    val_dataset_full = ImageDataset(df, transform=get_transformation(config, is_training=False))

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"Fold {fold+1}...")
        best_fold_acc = 0.0

        train_loader = get_dataloader(
            dataset=Subset(train_dataset_full, train_idx),
            batch_size=config["training"]["batch_size"],
            shuffle=True
        )

        val_loader = get_dataloader(
            dataset=Subset(val_dataset_full, val_idx),
            batch_size=config["training"]["batch_size"],
            shuffle=config["training"]["shuffle"]
        )

        model = get_trainable_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(config, model)
        
        for epoch in range(1, config["training"]["epochs"]+1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)

            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
        
        fold_results.append(best_fold_acc)
        
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    print(f"Final CV Results Accuracy: {np.mean(fold_results):.2f}%")

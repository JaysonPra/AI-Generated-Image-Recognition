import torch
from src.data.dataloader import get_dataloader
from torchvision import models, transforms
from torch import nn, optim
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    train_loader = get_dataloader(
        csv_file=config["csv_file"], 
        batch_size=config["training"]["batch_size"], 
        transform=_map_transformations(config),
        shuffle=config["training"]["shuffle"]
    )

    model = _get_trainable_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _get_optimizer(config, model)

    for epoch in range(1, config["training"]["epochs"]+1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch} complete. Avg loss: {running_loss / len(train_loader)}") 


def _get_trainable_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

def _get_optimizer(model, config):
    optimizer_name = config["training"]["optimizer"]
    optimizer_class = getattr(optim, optimizer_name)

    return optimizer_class(
        model.fc.parameters(),
        **config["training"]["optimizer_params"]
    )

def _map_transformations(config):
    augmentation_config = config["training"]["augmentations"]

    transformations_list = [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224)
    ]

    transformation_map = {
        "horizontalflip": transforms.RandomHorizontalFlip,
        "verticalflip": transforms.RandomVerticalFlip,
        "rotation": transforms.RandomRotation
    }

    for trans_name, trans_class in transformation_map.items():
        if trans_name in augmentation_config:
            params = augmentation_config[trans_name].get("parameters", {})

            transformations_list.append(trans_class(**params))

    transformations_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transforms.Compose(transformations_list)
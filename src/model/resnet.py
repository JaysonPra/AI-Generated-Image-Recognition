from torchvision import models
def get_trainable_model(config):
    """Get the trainable model of ResNet. The last layer is reset to 2 output neurons.

    Args:
        config (dict): The loaded YAML config file

    Returns:
        torchvision.models.Module: ResNet Model for training / testing
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    if not config["training"]["fine_tune"]:
        print("Freezing backbone...")
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    else:
        print("Unfreezing all layers...")

    return model
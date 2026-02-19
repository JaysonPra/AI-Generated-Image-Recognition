def get_trainable_model(config):
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
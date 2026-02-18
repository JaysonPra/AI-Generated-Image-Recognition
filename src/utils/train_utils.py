def get_optimizer(config, model):
    optimizer_name = config["training"]["optimizer"]
    optimizer_class = getattr(optim, optimizer_name)

    return optimizer_class(
        model.fc.parameters(),
        **config["training"]["optimizer_params"]
    )

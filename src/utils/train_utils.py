from torch import optim
def get_optimizer(config, model):
    optimizer_name = config["training"]["optimizer"]
    optimizer_class = getattr(optim, optimizer_name)

    if config["training"]["fine_tune"]:
        params_to_train = model.parameters()
    else:
        params_to_train = model.fc.parameters()

    return optimizer_class(
        params_to_train,
        **config["training"]["optimizer_params"]
    )

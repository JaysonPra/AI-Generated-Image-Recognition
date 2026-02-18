def get_transformation(config, is_training=True):
    transformations_list = [transforms.Resize((256, 256))]
    
    if is_training:
        augmentation_config = config["training"]["augmentations"]

        transformations_list.append(transforms.RandomResizedCrop(224))

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
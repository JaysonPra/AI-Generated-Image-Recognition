from torch.utils.data import DataLoader
from src.data.dataset import ImageDataset

def get_dataloader(csv_file, batch_size, shuffle, transform=None):
    """Function to get the Image DataLoader

    Args:
        csv_file (str): Train/Val/Test CSV file path
        transform (callable, optional): Optional transforms to be applied to the image. Defaults to None 
        batch_size (int): No. of images to load from the DataLoader per batch 
        shuffle (bool): Shuffling the images

    Returns:
        callable: The Image DataLoader
    """
    dataset = ImageDataset(
        csv_file=csv_file,
        transform=transform
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return loader
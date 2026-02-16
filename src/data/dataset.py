from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the train/val/test CSV file 
            transform (callable, optional): Optional transforms to be applied to a image. Defaults to None.
        """
        self.img_labels = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)


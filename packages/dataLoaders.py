from torchvision import datasets
from torch.utils.data import DataLoader
import os

def get_data_loader(path, transform, batch_size=64, shuffle=True):
    dataset = datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count()), dataset.classes
    
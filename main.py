import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import packages.transformData as transformData
import packages.dataLoaders as dataLoaders
from pathlib import Path

def main():
    """
    Main function.
    """
    train_transform = transformData.get_train_transform()
    test_transform = transformData.get_test_transform()
    dir_path = Path.cwd() / "data"
    dataloader_train = dataLoaders.get_data_loader(dir_path / "train", train_transform)
    dataloader_test = dataLoaders.get_data_loader(dir_path / "test", test_transform, shuffle=False)
    img,label  = next(iter(dataloader_train))
    print(f"img.shape {img.shape} -> [batch_size, channels, height, width]")
    print(f"label.shape {label.shape} -> [batch_size]") 



if __name__ == "__main__":
    main()
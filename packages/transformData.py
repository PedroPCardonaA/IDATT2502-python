import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from PIL import Image
import matplotlib.pyplot as plt

data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

def plot_transformed_images(image_path, transforms= data_transform, n=3, seed=42, saveFile = ""):
    if seed:
        torch.manual_seed(seed)
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    random_image_paths = random.sample(image_path_list, k = n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,7))
            ax[0].imshow(f)
            ax[0].set_title("Original Image {f.size}")
            ax[0].axis(False)

            transformed_image = transforms(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title("Transformed Image {transformed_image.shape}")
            ax[1].axis(False)

            fig.suptitle(f"Class: {image_path.parent.stem}", size=16)
            plt.savefig(saveFile / f"{image_path.stem}.png")



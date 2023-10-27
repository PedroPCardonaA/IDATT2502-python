import os
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import packages.transformData as transformData
from torchinfo import summary
import io
import sys


plot_path = Path.cwd() / "plots"

def walk_trough_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  """
  for dirpath,dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def visualize_random_image(dir_path):
    image_path_list = list(dir_path.glob("*/*/*.jpg"))
    random_image_path = random.choice(image_path_list)
    image_class = random_image_path.parent.stem
    img = Image.open(random_image_path)
    img_as_array = np.array(img)
    plt.figure(figsize=(10,7))
    plt.imshow(img_as_array)
    plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color channels]]")
    plt.axis(False)
    plt.savefig(plot_path / f"{image_class}.png")
    
def save_summary(model, input_size, filename):
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    summary(model, input_size=input_size)
    summary_str = output_buffer.getvalue()
    sys.stdout = sys.__stdout__

    out_fol = Path.cwd() / "plots"
    out_fol.mkdir(exist_ok=True)
    with open(out_fol / filename, "w") as f:
        f.write(summary_str)
    
    

def main():
    """
    Main function.
    """
    dir_path = Path.cwd() / "data"
    print(f"Current working directory is {dir_path}")
    #walk_trough_dir(dir_path)
    visualize_random_image(dir_path)
    transformData.plot_transformed_images(dir_path, saveFile=Path.cwd() / "plots")

if __name__ == "__main__":
    main()
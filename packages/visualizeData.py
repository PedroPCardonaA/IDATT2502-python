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
from typing import Dict, List
from torchmetrics import ConfusionMatrix

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
    

def plot_loss_curves(results: Dict[str, List[float]], number:int = 1):
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  accuracy = results["train_acc"]
  test_accuracy=results["test_acc"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15,7))

  plt.subplot(1,2,1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")

  plt.subplot(1,2,1)
  plt.plot(epochs,accuracy, label = "train_accuracy")
  plt.plot(epochs,test_accuracy, label="test_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()
  plt.savefig('../plots/'+ str(number)+'.jpg', bbox_inches='tight')   


def plot_confusion_matrix(model, dataloader, class_names, device):
    
    confusion_matrix = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    model.eval()

    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Assuming classification is along dim 1

            predicted_labels.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    # Convert indices to class labels
    predicted_tensor = torch.tensor(predicted_labels)
    true_tensor = torch.tensor(true_labels)

    # Update the confusion matrix with predicted and true labels (as tensors)
    confusion_matrix.update(predicted_tensor, true_tensor)

    # Compute the confusion matrix
    matrix = confusion_matrix.compute()
    matrix_np = matrix.numpy()

    # Normalize the matrix by dividing each value by the column sum
    normalized_matrix = matrix_np / matrix_np.sum(axis=0, keepdims=True)

    # Create a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Create a heatmap using seaborn with the custom colormap
    plt.figure(figsize=(10, 7))
    sns.heatmap(normalized_matrix, annot=True, cmap=cmap)

    # Set class names as labels on x and y axes
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=45)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=0)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix based on Column Sums")
    plt.show()


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
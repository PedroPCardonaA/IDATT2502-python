import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import packages.transformData as transformData
import packages.dataLoaders as dataLoaders
import packages.models as models
import packages.visualizeData as visualizeData
from pathlib import Path
import packages.loops as loops
from timeit import default_timer as timer

def main():
    """
    Main function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transformData.get_train_transform()
    test_transform = transformData.get_test_transform()
    dir_path = Path.cwd() / "data"
    dataloader_train, classes = dataLoaders.get_data_loader(dir_path / "train", train_transform)
    dataloader_test = dataLoaders.get_data_loader(dir_path / "test", test_transform, shuffle=False)
    img,label  = next(iter(dataloader_train))
    print(f"img.shape {img.shape} -> [batch_size, channels, height, width]")
    print(f"label.shape {label.shape} -> [batch_size]") 
    print(f"classes: {classes}")

    torch.manual_seed(42)
    model = models.garbage_classifier(input_shape=3, hidden_units=64, output_shape=len(classes)).to(device)
    print(model)

    image_batch, label_batch = next(iter(dataloader_train))
    result = model(image_batch.to(device))
    print(result)
    print(result.shape)

    visualizeData.save_summary(model, (1,3,224,224), "model_0_summary.txt")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    NUM_EPOCHS = 5

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start = timer()

    model_0_results = loops.train(model, 
    dataloader_train, 
    dataloader_test, 
    loss_fn, 
    optimizer, 
    NUM_EPOCHS, 
    device)

    end = timer()
    print(f"Training time: {end - start:.3f}s")

    




if __name__ == "__main__":
    main()
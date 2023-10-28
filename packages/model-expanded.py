import torch
from torch import nn

class garbage_classifier_more_conv(nn.Module):
    def __init__(self, input_shape:int,
    hidden_units:int,
    output_shape:int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
            stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
            stride=2)
        )
        
        #Additional convolutional block to compare metrics to the standard model
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
            stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*28*28, 
            out_features=output_shape),
        )

    def forward(self, x):
        return self.classifier(self.conv_block3(self.conv_block_2(self.conv_block_1(x))))

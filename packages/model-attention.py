import torch
from torch import nn

class garbage_classifier_attention(nn.Module):
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
        
        #Added attention mechanism to compare metrics to the standard model
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*56*56,
            out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        
        #Attention Mechanism to focus on important features
        att = self.attention(x)
        x = x * att 
        
        return self.classifier(x)

import torch
from torch import nn

class garbage_classifier(nn.Module):
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
            nn.Conv2d(in_channels=hidden_units,
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
            nn.Conv2d(in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
            stride=2)
        )

        # TODO: Maybe add more conv blocks

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*56*56,
            out_features=output_shape),
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

        
class garbage_classifier_5L_attention(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units * 4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(hidden_units * 4, hidden_units * 4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(hidden_units * 4, hidden_units * 4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(hidden_units * 4, hidden_units * 4, 1, 1, 0),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 4 * 7 * 7, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_5(self.conv_block_4(self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))))
        return self.classifier(x * self.attention(x))

class garbage_classifier_5L_attention_with_batch_and_dropout(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(hidden_units)  # Adding Batch Normalization
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(hidden_units * 2)  # Adding Batch Normalization
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units * 4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(hidden_units * 4)  # Adding Batch Normalization
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(hidden_units * 4, hidden_units * 4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(hidden_units * 4)  # Adding Batch Normalization
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(hidden_units * 4, hidden_units * 4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(hidden_units * 4)  # Adding Batch Normalization
        )

        self.dropout = nn.Dropout(0.5)  # Adding Dropout for regularization

        self.attention = nn.Sequential(
            nn.Conv2d(hidden_units * 4, hidden_units * 4, 1, 1, 0),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 4 * 7 * 7, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_5(self.conv_block_4(self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))))
        x = self.dropout(x)  # Applying Dropout
        return self.classifier(x * self.attention(x))
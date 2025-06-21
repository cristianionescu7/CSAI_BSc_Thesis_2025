import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Defines the deep learning models used for EEG-based inner speech decoding:

- CNN: A baseline convolutional neural network that captures spatial and 
  short-term temporal patterns across EEG channels and timepoints based
  on the EEGNet architecture (Lawhern et al., 2018).

- LSTM_CNN: A hybrid model that first applies an LSTM layer to encode 
  temporal dynamics across time, followed by a CNN that extracts spatial 
  features from the LSTM outputs.

These models are designed to operate on EEG data formatted as 
(trials, channels, samples) and support PyTorch training and 
evaluation pipelines.
"""



class CNN(nn.Module):
    def __init__(self, num_channels=128, num_samples=1153, num_classes=4):
        super(CNN, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(num_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25)
        )

        # Dynamically compute the flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_channels, num_samples)
            dummy_output = self._forward_features(dummy_input)
            self.feature_dim = dummy_output.shape[1]

        # Two fully connected layers
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.dropout_fc = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_features(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)  # flatten
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

class LSTM_CNN(nn.Module):
    def __init__(self, num_channels, num_samples, hidden_size=64, num_classes=4):
        super(LSTM_CNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=num_channels,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.dropout_lstm = nn.Dropout(0.2)
        direction_factor = 2 if self.lstm.bidirectional else 1
        
        # CNN: [B, 1, H, T]
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=8, 
                               kernel_size=(1, 64), 
                               padding=(0, 32), 
                               bias = False)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(in_channels=8, 
                               out_channels=16, 
                               kernel_size=(hidden_size, 1), 
                               groups=8, 
                               bias = False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=(1, 16), 
                               padding=(0, 8), 
                               bias = False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout3 = nn.Dropout(0.25)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, hidden_size * direction_factor, num_samples)
            x = self.bn1(self.conv1(dummy_input))
            x = self.pool2(self.bn2(self.conv2(x)))
            x = self.pool3(self.bn3(self.conv3(x)))
            self.flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout_fc = nn.Dropout(0.15)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        # x: [B, C, T] → [B, T, C] for LSTM
        x = x.permute(0, 2, 1)  # [B, 1153, 128]

        lstm_out, _ = self.lstm(x)  # [B, T, H]
        lstm_out = self.dropout_lstm(lstm_out)

        cnn_input = lstm_out.permute(0, 2, 1).unsqueeze(1)  # [B, 1, H, T]

        x = self.bn1(self.conv1(cnn_input))

        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x
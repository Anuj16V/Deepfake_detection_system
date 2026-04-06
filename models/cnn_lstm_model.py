import torch
import torch.nn as nn
import torchvision.models as models


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()#This initializes the parent class.

        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier = nn.Identity()

        # Freeze all layers first
        for param in self.cnn.parameters():
            param.requires_grad = False

# Unfreeze last convolution block
        for param in self.cnn.features[-1].parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        final_feature = lstm_out[:, -1, :]
        output = self.fc(final_feature)

        return output.squeeze()
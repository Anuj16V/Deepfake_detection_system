import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from models.cnn_lstm_model import CNN_LSTM
from scripts.dataset import DeepfakeSequenceDataset

device = torch.device("cpu")

dataset = DeepfakeSequenceDataset()
loader = DataLoader(dataset, batch_size=8)

model = CNN_LSTM()
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
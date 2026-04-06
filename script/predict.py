import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image

from models.cnn_lstm_model import CNN_LSTM

MODEL_PATH = "deepfake_model.pth"
SEQ_LEN = 20
device = torch.device("cpu")

# Load model
model = CNN_LSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // SEQ_LEN,1)

    count = 0
    frame_id = 0

    while cap.isOpened() and count < SEQ_LEN:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            image = transform(image)
            frames.append(image)

            count += 1

        frame_id += 1

    cap.release()

    frames = torch.stack(frames)
    frames = frames.unsqueeze(0)

    return frames


def predict(video_path):

    frames = extract_frames(video_path)
    frames = frames.to(device)

    with torch.no_grad():
        output = model(frames)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        print(f"\nPrediction: FAKE ({prob*100:.2f}%)")
    else:
        print(f"\nPrediction: REAL ({(1-prob)*100:.2f}%)")


video = input("Enter video path: ")
predict(video)
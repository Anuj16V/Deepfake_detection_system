import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
SEQUENCE_PATH = "deepfake_detection/data/sequences"
SEQ_LEN = 20

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],#normalize Since we are using a pretrained MobileNetV2,  it expects images normalized like ImageNet dataset.  #
                         [0.229, 0.224, 0.225])
])

class DeepfakeSequenceDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.labels = []

        for label, class_name in enumerate(["real", "fake"]):
            class_path = os.path.join(SEQUENCE_PATH, class_name)

            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                self.samples.append(video_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]

        frames = sorted(os.listdir(video_path))[:SEQ_LEN]
        images = []

        for frame in frames:
            img_path = os.path.join(video_path, frame)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            images.append(img)

        images = torch.stack(images)

        return images, torch.tensor(label, dtype=torch.float32)
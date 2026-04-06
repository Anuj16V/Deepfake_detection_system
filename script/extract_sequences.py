import os
import cv2
from tqdm import tqdm

SELECTED_PATH = "deepfake_detection/data/selected"
SEQUENCE_PATH = "deepfake_detection/data/sequences"

FRAMES_PER_VIDEO = 20
IMG_SIZE = 224


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < FRAMES_PER_VIDEO:
        cap.release()
        return

    interval = total_frames // FRAMES_PER_VIDEO

    frame_count = 0
    saved_count = 0

    while cap.isOpened() and saved_count < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            filename = os.path.join(output_folder, f"{saved_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()


def process_class(class_name):
    input_folder = os.path.join(SELECTED_PATH, class_name)
    output_class_folder = os.path.join(SEQUENCE_PATH, class_name)

    os.makedirs(output_class_folder, exist_ok=True)

    videos = [v for v in os.listdir(input_folder) if v.endswith(".mp4")]

    for video in tqdm(videos, desc=f"Processing {class_name}"):
        video_path = os.path.join(input_folder, video)
        video_name = os.path.splitext(video)[0]
        output_folder = os.path.join(output_class_folder, video_name)

        os.makedirs(output_folder, exist_ok=True)

        extract_frames(video_path, output_folder)


process_class("real")
process_class("fake")

print("✅ Frame sequence extraction complete!")
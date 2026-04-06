import os
import random
import shutil

# Correct paths based on your project structure
RAW_PATH = "deepfake_detection/data/raw"
SELECTED_PATH = "deepfake_detection/data/selected"

NUM_VIDEOS = 400  # number per class

random.seed(42)


def select_videos(class_name):
    src_folder = os.path.join(RAW_PATH, class_name)
    dst_folder = os.path.join(SELECTED_PATH, class_name)

    os.makedirs(dst_folder, exist_ok=True)

    # Get only video files
    videos = [v for v in os.listdir(src_folder) if v.endswith(".mp4")]

    if len(videos) < NUM_VIDEOS:
        print(f"⚠ Not enough videos in {class_name}")
        return

    selected = random.sample(videos, NUM_VIDEOS)

    for video in selected:
        shutil.copy(
            os.path.join(src_folder, video),
            os.path.join(dst_folder, video)
        )

    print(f"✅ {class_name}: {len(selected)} videos selected")


select_videos("real")
select_videos("fake")

print("🎉 Dataset selection complete!")
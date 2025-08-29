import os
import shutil
import random

# ==== Configuration ====
SOURCE_DIR = "kvasir-v2"
DEST_DIR = "CLASSIFICATION_DATASET"
LABEL_MAP = {
    "malignant": ["polyps", "dyed-lifted-polyps", "dyed-resection-margins"],
    "non_malignant": ["normal-cecum", "normal-pylorus", "normal-z-line", "esophagitis", "ulcerative-colitis"]
}
SPLIT_RATIO = 0.8  # 80% train, 20% test

# ==== Create Directory Structure ====
for split in ["train", "test"]:
    for label in LABEL_MAP:
        os.makedirs(os.path.join(DEST_DIR, split, label), exist_ok=True)

# ==== Process and Split ====
for label, folders in LABEL_MAP.items():
    all_images = []
    for folder in folders:
        folder_path = os.path.join(SOURCE_DIR, folder)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        all_images.extend(files)

    random.shuffle(all_images)
    split_idx = int(len(all_images) * SPLIT_RATIO)
    train_files = all_images[:split_idx]
    test_files = all_images[split_idx:]

    for f in train_files:
        shutil.copy2(f, os.path.join(DEST_DIR, "train", label, os.path.basename(f)))
    for f in test_files:
        shutil.copy2(f, os.path.join(DEST_DIR, "test", label, os.path.basename(f)))

print("✅ Dataset split into 'train' and 'test' in:", DEST_DIR)


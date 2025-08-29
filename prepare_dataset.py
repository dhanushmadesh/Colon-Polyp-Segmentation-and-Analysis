import os
import cv2
import shutil
import random
from tqdm import tqdm

image_dir = 'Kvasir-SEG/images'
mask_dir = 'Kvasir-SEG/masks'
output_base = 'dataset'
img_size = (256, 256)
train_ratio = 0.8

# Create output folders
for split in ['train', 'val']:
    for folder in ['images', 'masks']:
        os.makedirs(os.path.join(output_base, split, folder), exist_ok=True)

# Match files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
paired = []

for img_file in image_files:
    index = img_file.split('_')[-1]
    mask_file = f"mask_{index}"
    if os.path.exists(os.path.join(mask_dir, mask_file)):
        paired.append((img_file, mask_file))

random.shuffle(paired)
split_idx = int(len(paired) * train_ratio)
train_pairs = paired[:split_idx]
val_pairs = paired[split_idx:]

def process_pair(pair_list, split):
    for img_name, mask_name in tqdm(pair_list, desc=f"Processing {split}"):
        img = cv2.imread(os.path.join(image_dir, img_name))
        img = cv2.resize(img, img_size)
        cv2.imwrite(os.path.join(output_base, split, 'images', img_name), img)

        mask = cv2.imread(os.path.join(mask_dir, mask_name), 0)
        mask = cv2.resize(mask, img_size)
        cv2.imwrite(os.path.join(output_base, split, 'masks', mask_name), mask)

process_pair(train_pairs, 'train')
process_pair(val_pairs, 'val')

print("✅ Dataset prepared in 'dataset/' folder")

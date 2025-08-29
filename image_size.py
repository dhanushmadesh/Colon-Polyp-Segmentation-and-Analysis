import os
from PIL import Image

folder = "CLASSIFICATION_DATASET/train/malignant"  # or any folder in your dataset
sizes = []

for fname in os.listdir(folder):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            with Image.open(os.path.join(folder, fname)) as img:
                sizes.append(img.size)
        except Exception as e:
            print(f"⚠️ Failed to open {fname}: {e}")

# Show the most common sizes
from collections import Counter
common_sizes = Counter(sizes).most_common(10)
print("🖼️ Most common image sizes:")
for size, count in common_sizes:
    print(f"{size[0]}x{size[1]}  →  {count} images")

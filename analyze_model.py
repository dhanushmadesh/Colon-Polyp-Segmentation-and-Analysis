import os
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from glob import glob
from sklearn.metrics import precision_recall_curve, auc

# === Paths ===
MODEL_PATH = "model/best_model_b4.h5"
DATASET_TEST_IMG = "dataset/val/images"   # update if needed
DATASET_TEST_MASK = "dataset/val/masks"
SAVE_DIR = "model_evaluation"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "predictions"), exist_ok=True)

# === Load Model ===
sm.set_framework("tf.keras")
sm.framework()

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "dice_loss": sm.losses.DiceLoss(),
        "iou_score": sm.metrics.IOUScore(threshold=0.5),
        "f1-score": sm.metrics.FScore(threshold=0.5),
    },
)

# === Save Model Summary ===
with open(os.path.join(SAVE_DIR, "model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# === Dataset Loader ===
IMG_HEIGHT, IMG_WIDTH = 256, 256

def load_images_and_masks(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.jpg")))
    images, masks = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        images.append(img)

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)
    return np.array(images), np.array(masks)

print("🔄 Loading validation/test data...")
test_images, test_masks = load_images_and_masks(DATASET_TEST_IMG, DATASET_TEST_MASK)
print("✅ Test set:", test_images.shape, test_masks.shape)

# === Compile Model for Evaluation ===
dice_loss = sm.losses.DiceLoss()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=dice_loss, metrics=metrics)

# === Evaluate Model ===
print("📊 Evaluating model...")
scores = model.evaluate(test_images, test_masks, verbose=1)
metrics_names = ["Loss", "IoU", "F1-score"]

with open(os.path.join(SAVE_DIR, "evaluation_metrics.txt"), "w") as f:
    for name, score in zip(metrics_names, scores):
        f.write(f"{name}: {score:.4f}\n")
        print(f"{name}: {score:.4f}")

# === Predictions ===
print("🎨 Generating predictions...")
preds = model.predict(test_images)

# === Precision-Recall Curve ===
y_true = (test_masks.flatten() > 0.5).astype(int)  # binarize GT
y_scores = preds.flatten()                         # probabilities
precision, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "precision_recall_curve.png"))
plt.close()

# === Save sample predictions (Original | GT Mask | Predicted Overlay) ===
print("🖼 Saving qualitative results...")
for i in range(min(5, len(test_images))):
    original = (test_images[i] * 255).astype(np.uint8)
    mask_true = (test_masks[i].squeeze() * 255).astype(np.uint8)
    mask_pred = (preds[i].squeeze() > 0.5).astype(np.uint8) * 255

    # Overlay predicted mask
    overlay = original.copy()
    overlay[mask_pred > 127] = [255, 0, 0]  # red overlay
    blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

    out_img = np.hstack([
        original,
        cv2.cvtColor(mask_true, cv2.COLOR_GRAY2BGR),
        blended
    ])
    save_path = os.path.join(SAVE_DIR, "predictions", f"sample_{i+1}.jpg")
    cv2.imwrite(save_path, out_img)

print(f"✅ All results saved inside {SAVE_DIR}/")

# === Save a final summary report ===
with open(os.path.join(SAVE_DIR, "report.txt"), "w") as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("=======================\n\n")
    f.write(f"Model Path: {MODEL_PATH}\n")
    f.write(f"Test Data: {DATASET_TEST_IMG}, {DATASET_TEST_MASK}\n\n")
    for name, score in zip(metrics_names, scores):
        f.write(f"{name}: {score:.4f}\n")
    f.write(f"Precision-Recall AUC: {pr_auc:.4f}\n\n")
    f.write("Qualitative results saved in ./model_evaluation/predictions/\n")


import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm  # ✅ Installed from GitHub

# ==== Config ====
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 8
EPOCHS = 40
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# ==== Dataset paths ====
train_img_dir = "dataset/train/images"
train_mask_dir = "dataset/train/masks"
val_img_dir = "dataset/val/images"
val_mask_dir = "dataset/val/masks"

# ==== Load Data ====
def load_images_and_masks(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.jpg")))

    images, masks = [], []
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        images.append(img)

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)

    return np.array(images), np.array(masks)

print("🔄 Loading data...")
train_images, train_masks = load_images_and_masks(train_img_dir, train_mask_dir)
val_images, val_masks = load_images_and_masks(val_img_dir, val_mask_dir)

print("✅ Training set:", train_images.shape)
print("✅ Validation set:", val_images.shape)

# ==== Set Framework ====
sm.set_framework('tf.keras')
sm.framework()

# ==== Define Model ====
model = sm.Unet(  # ✅ Use Unet, not UnetPlusPlus
    backbone_name='efficientnetb4',
    input_shape=INPUT_SHAPE,
    classes=1,
    activation='sigmoid',
    encoder_weights='imagenet'
)

# ==== Compile ====
dice_loss = sm.losses.DiceLoss()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=metrics)
model.summary()

# ==== Callbacks ====
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "model/best_model_b4.h5", save_best_only=True, monitor='val_loss', mode='min', verbose=1)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

# ==== Train ====
print("🚀 Training overnight model...")
model.fit(train_images, train_masks,
          validation_data=(val_images, val_masks),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          callbacks=[checkpoint_cb, earlystop_cb])

print("✅ Done. Best model saved as model/best_model_b4.h5")

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import segmentation_models as sm
from skimage.measure import label, regionprops
import pandas as pd
import zipfile
import io
import os
import traceback

# === Load Model ===
sm.set_framework("tf.keras")
sm.framework()
model = tf.keras.models.load_model(
    "model/best_model_b4.h5",
    compile=False,
    custom_objects={
        "dice_loss": sm.losses.DiceLoss(),
        "iou_score": sm.metrics.IOUScore(threshold=0.5),
        "f1-score": sm.metrics.FScore(threshold=0.5),
    },
)

# === Helpers ===
def overlay_mask_on_image(original, mask, alpha=0.4):
    original_np = np.array(original.convert("RGB"))
    mask_resized = np.array(mask.resize(original.size))
    overlay_color = [255, 0, 0]
    faint_green = np.array([100, 180, 100])
    color_mask = np.zeros_like(original_np)
    color_mask[mask_resized <= 127] = faint_green
    color_mask[mask_resized > 127] = overlay_color
    overlay = cv2.addWeighted(original_np, 0.6, color_mask, alpha, 0)
    return Image.fromarray(overlay)

def analyze_mask(mask_array, threshold=100):
    labeled_mask = label(mask_array)
    props = regionprops(labeled_mask)
    total_area, count = 0, 0
    shape_features = []
    for prop in props:
        if prop.area < threshold:
            continue
        count += 1
        total_area += prop.area
        shape_features.append({
            "Area": prop.area,
            "Perimeter": round(prop.perimeter, 2),
            "Solidity": round(prop.solidity, 3),
            "Eccentricity": round(prop.eccentricity, 3),
            "Aspect Ratio": round((prop.bbox[2] - prop.bbox[0]) / max(1, (prop.bbox[3] - prop.bbox[1])), 3),
        })
    return total_area, count, shape_features

def generate_polyp_description(dominant_feat, coverage, confidence, count):
    if dominant_feat is None or count == 0:
        return "No polyp detected in this image."
    if dominant_feat["Area"] < 500:
        size_txt = "small"
    elif dominant_feat["Area"] < 2000:
        size_txt = "medium-sized"
    else:
        size_txt = "large"
    if dominant_feat["Eccentricity"] < 0.5:
        shape_txt = "round"
    elif dominant_feat["Eccentricity"] < 0.8:
        shape_txt = "oval-shaped"
    else:
        shape_txt = "elongated or flat"
    if dominant_feat["Solidity"] > 0.9:
        border_txt = "smooth, regular borders"
    elif dominant_feat["Solidity"] > 0.7:
        border_txt = "slightly irregular borders"
    else:
        border_txt = "irregular, serrated borders"
    conf_txt = "high confidence" if confidence > 80 else "moderate confidence" if confidence > 50 else "low confidence"
    base = f"{size_txt} {shape_txt} polyp with {border_txt} (~{coverage:.2f}% coverage, {conf_txt})"
    return f"{count} polyps detected. Dominant region: {base}." if count > 1 else f"This image shows a {base}."

# === Core Processing (aggregated: ONE numeric row per image) ===
def process_images(files):
    numeric_rows, summary_rows, image_rows = [], [], []
    masks_zip = io.BytesIO()
    with zipfile.ZipFile(masks_zip, "w") as zf:
        for file in files:
            image = Image.open(file).convert("RGB")
            img_resized = image.resize((256, 256))
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr)[0]
            mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask).resize(image.size)

            total_area, count, shape_features = analyze_mask(np.array(mask_img))
            image_area = image.size[0] * image.size[1]
            coverage = (total_area / image_area) * 100 if image_area > 0 else 0.0
            polyp_pixels = pred[pred > 0.5]
            confidence_score = float(np.mean(polyp_pixels) * 100) if polyp_pixels.size else 0.0

            # save mask
            name_wo_ext, _ = os.path.splitext(os.path.basename(file))
            mask_filename = f"{name_wo_ext}_mask.jpg"
            img_bytes = io.BytesIO(); mask_img.save(img_bytes, format="JPEG")
            zf.writestr(mask_filename, img_bytes.getvalue())

            dominant = max(shape_features, key=lambda f: f["Area"]) if shape_features else None
            if total_area > 0 and shape_features:
                weighted_solidity = round(sum(f["Solidity"] * f["Area"] for f in shape_features) / total_area, 3)
                weighted_ecc = round(sum(f["Eccentricity"] * f["Area"] for f in shape_features) / total_area, 3)
            else:
                weighted_solidity, weighted_ecc = 0.0, 0.0

            # numeric row (exact order)
            if count == 0:
                numeric_rows.append({
                    "Image Name": os.path.basename(file),
                    "Polyp Count": 0,
                    "Polyp Area (px)": 0,
                    "Solidity": 0,
                    "Eccentricity": 0,
                    "% Coverage": f"{coverage:.2f}",
                    "Aspect Ratio": 0,
                    "Confidence (%)": f"{confidence_score:.2f}",
                })
                summary_rows.append({"Image Name": os.path.basename(file), "Description": "No polyp detected in this image."})
            else:
                numeric_rows.append({
                    "Image Name": os.path.basename(file),
                    "Polyp Count": count,
                    "Polyp Area (px)": int(total_area),
                    "Solidity": weighted_solidity,
                    "Eccentricity": weighted_ecc,
                    "% Coverage": f"{coverage:.2f}",
                    "Aspect Ratio": dominant["Aspect Ratio"] if dominant else 0,
                    "Confidence (%)": f"{confidence_score:.2f}",
                })
                summary_rows.append({
                    "Image Name": os.path.basename(file),
                    "Description": generate_polyp_description(dominant, coverage, confidence_score, count),
                })

            image_rows.append([
                [image, f"{os.path.basename(file)} - Original"],
                [overlay_mask_on_image(image, mask_img), f"{os.path.basename(file)} - Overlay"],
                [mask_img, f"{os.path.basename(file)} - Mask"],
            ])

    masks_zip.seek(0)
    numeric_cols = [
        "Image Name", "Polyp Count", "Polyp Area (px)", "Solidity",
        "Eccentricity", "% Coverage", "Aspect Ratio", "Confidence (%)",
    ]
    df_numeric = pd.DataFrame(numeric_rows, columns=numeric_cols)
    df_summary = pd.DataFrame(summary_rows, columns=["Image Name", "Description"])

    csv_path = "polyp_summary.csv"; df_numeric.to_csv(csv_path, index=False)
    zip_path = "masks.zip"; open(zip_path, "wb").write(masks_zip.getvalue())
    return image_rows, df_numeric, df_summary, csv_path, zip_path

def run_segmentation(files):
    try:
        image_rows, df_numeric, df_summary, csv_path, zip_path = process_images(files)
        gallery_data = []; [gallery_data.extend(row) for row in image_rows]
        return gallery_data, df_numeric, df_summary, csv_path, zip_path
    except Exception as e:
        traceback.print_exc()
        return [], pd.DataFrame([{"Error": str(e)}]), pd.DataFrame([{"Error": str(e)}]), None, None

# === CSS: make Summary table first column narrow & Description wide ===
css = """
.full-width-upload { width: 100% !important; margin: 10px auto !important; padding: 20px !important;
    background-color: #E3F2FD !important; border-radius: 10px !important; }
.full-width-button { width: 100% !important; font-size: 18px !important; font-weight: bold !important;
    padding: 15px !important; margin: 15px auto !important; background-color: #1976D2 !important;
    color: white !important; border-radius: 8px !important; }

/* Summary table layout */
#summary-table table { table-layout: fixed; width: 100%; }
#summary-table table th:nth-child(1),
#summary-table table td:nth-child(1) { width: 240px !important; }
#summary-table table th:nth-child(2),
#summary-table table td:nth-child(2) { width: calc(100% - 240px) !important; }

/* Wrap description nicely and tighten spacing */
#summary-table table td:nth-child(2) { white-space: pre-wrap !important; word-break: break-word !important; }
#summary-table table td, #summary-table table th { padding: 8px 10px !important; vertical-align: top !important; line-height: 1.25 !important; }

/* Optional: also prevent the filename column from wrapping to multiple lines */
#summary-table table td:nth-child(1) { white-space: nowrap !important; overflow: hidden; text-overflow: ellipsis; }
"""

# === Gradio UI ===
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("<h1 style='text-align:center; color:#1565C0;'>🩺 Colon Polyp Segmentation & Morphological Analysis</h1>")
    gr.Markdown(
        """
        ### ℹ️ What is this?
        A Deep Learning tool using **EfficientNetB4 + UNet** to detect and segment colon polyps in colonoscopy images.  
        Provides **objective numeric analysis** and a **natural-language summary** per image.

        ⚠️ *Disclaimer:* Descriptions are AI-generated from image morphology and are **not a clinical diagnosis**.
        """
    )

    file_input = gr.File(file_types=[".jpg", ".jpeg"], type="filepath",
                         label="📤 Step 1: Upload Colonoscopy Images",
                         file_count="multiple", elem_classes="full-width-upload")
    run_btn = gr.Button("🚀 Step 2: Run Segmentation", variant="primary", elem_classes="full-width-button")

    gallery = gr.Gallery(label="Results (Original | Overlay | Mask)", columns=3, rows=5, show_label=True)

    results_table = gr.Dataframe(
        headers=[
            "Image Name", "Polyp Count", "Polyp Area (px)", "Solidity",
            "Eccentricity", "% Coverage", "Aspect Ratio", "Confidence (%)",
        ],
        interactive=False, wrap=True, label="📊 Numeric Analysis (One Row per Image)",
        elem_id="numeric-table"
    )

    # ⬇️ Summary table gets the special CSS via elem_id
    summary_table = gr.Dataframe(
        headers=["Image Name", "Description"], interactive=False, wrap=True,
        label="🧾 Summary (Per Image)", elem_id="summary-table"
    )

    with gr.Row():
        csv_download = gr.File(label="📊 Download Numeric CSV", type="filepath")
        zip_download = gr.File(label="🗂️ Download Masks ZIP", type="filepath")

    run_btn.click(fn=run_segmentation, inputs=file_input,
                  outputs=[gallery, results_table, summary_table, csv_download, zip_download])

    gr.Markdown(
        """
        ---
        ✅ Developed by **Dhanush, Hemanth, Sathya, Likhith** | Major Project – ISE  
        🚀 Powered by TensorFlow & EfficientNetB4 + UNet  
        """
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)

import streamlit as st
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from zipfile import ZipFile
import cv2
from skimage.measure import label, regionprops
import pandas as pd


def load_model(model_path):
    import segmentation_models as sm
    sm.set_framework('tf.keras')
    sm.framework()
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                'dice_loss': sm.losses.DiceLoss(),
                'iou_score': sm.metrics.IOUScore(threshold=0.5),
                'f1-score': sm.metrics.FScore(threshold=0.5)
            }
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def clear_temp_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


def generate_data_batches(data_dir, batch_size, target_size, seed=None):
    rescale_factor = 1.0 / 255.0
    if seed is not None:
        np.random.seed(seed)
    test_data_generator = ImageDataGenerator(rescale=rescale_factor)
    return test_data_generator.flow_from_directory(
        data_dir + '/app_images',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
        seed=seed
    )


# New: Area-aware polyp count estimation (range)
def analyze_mask(mask_array):
    labeled_mask = label(mask_array)
    props = regionprops(labeled_mask)
    areas = [prop.area for prop in props]
    total_area = sum(areas)

    min_count = sum(1 for a in areas if 300 < a < 8000)         # likely individual polyps
    likely_multiple = sum(1 for a in areas if a >= 8000)        # possibly merged blobs
    estimated_min = min_count + likely_multiple
    estimated_max = min_count + (2 * likely_multiple)
    count_range = f"{estimated_min} – {estimated_max}"

    return total_area, count_range, areas


def get_severity_tag(area, count_range_text):
    if area == 0:
        return "✅ No Polyp Detected", 0
    elif area < 3000:
        return "🟢 Small Polyp", 1
    elif area < 8000:
        return "🟡 Medium Polyp", 2
    else:
        return "🔴 Large or Multiple Polyps", 3


def overlay_mask_on_image(original, mask, severity_text, alpha=0.4):
    original_np = np.array(original.convert('RGB'))
    mask_resized = np.array(mask.resize(original.size))

    # Define polyp color mapping
    color_map = {
        "🟢 Small Polyp": [0, 255, 0],
        "🟡 Medium Polyp": [255, 255, 0],
        "🔴 Large or Multiple Polyps": [255, 0, 0]
    }

    overlay_color = color_map.get(severity_text, [255, 0, 0])  # default red if not found

    # Create a color mask for both polyp and non-polyp regions
    color_mask = np.zeros_like(original_np)

    # 1️⃣ Polyp regions
    color_mask[mask_resized > 127] = overlay_color

    # 2️⃣ Non-polyp background
    faint_green = np.array([100, 180, 100])  # visible green
    color_mask[mask_resized <= 127] = faint_green

    # 3️⃣ Blend with original
    overlay = cv2.addWeighted(original_np, 0.6, color_mask, 0.4, 0)

    return Image.fromarray(overlay)


def get_risk_level(area, count_range_text):
    if area == 0:
        return "🟢 Low Risk"
    est_min = int(count_range_text.split("–")[0].strip())
    if area > 8000 or est_min >= 2:
        return "🔴 High Risk"
    elif area >= 3000:
        return "🟡 Moderate Risk"
    else:
        return "🟢 Low Risk"


def process_and_display_images(loaded_model, uploaded_files, temp_dir):
    try:
        clear_temp_directory(temp_dir)
        masks_dir = os.path.join(temp_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        uploaded_files_sorted = sorted(uploaded_files, key=lambda x: x.name)
        for uploaded_file in uploaded_files_sorted:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        batch_size = 16
        test_images_gen = generate_data_batches('./temp_app_data/', batch_size, (256, 256), seed=123)
        total_files = len(uploaded_files_sorted)
        steps_needed = np.ceil(total_files / batch_size)
        global_index = 0

        results_summary = []

        for _ in range(int(steps_needed)):
            batch = next(test_images_gen)
            predictions = loaded_model.predict(batch)
            binary_predictions = (predictions > 0.5).astype(np.uint8)

            for j, pred_image in enumerate(binary_predictions):
                if global_index >= len(uploaded_files_sorted):
                    break

                pred_pil_image = Image.fromarray(pred_image.squeeze() * 255)
                polyp_pixels = pred_image[pred_image > 0.5]
                confidence_score = np.mean(polyp_pixels) * 100 if polyp_pixels.size else 0
                original_image = Image.open(uploaded_files_sorted[global_index])
                pred_pil_image = pred_pil_image.resize(original_image.size)

                area, count_range, _ = analyze_mask(np.array(pred_pil_image))
                image_area = pred_pil_image.size[0] * pred_pil_image.size[1]
                coverage = (area / image_area) * 100 if image_area > 0 else 0
                severity_text, severity_score = get_severity_tag(area, count_range)
                risk_level = get_risk_level(area, count_range)
                overlay_image = overlay_mask_on_image(original_image, pred_pil_image, severity_text)

                original_file_name = os.path.basename(uploaded_files_sorted[global_index].name)
                name_wo_ext = os.path.splitext(original_file_name)[0]

                st.markdown(f"<h3 style='color:#1565C0;'>🖼 Image: {original_file_name}</h3>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(original_image, caption='Original Image', use_container_width=True)
                with col2:
                    st.image(overlay_image, caption='Overlay Mask', use_container_width=True)
                with col3:
                    st.image(pred_pil_image, caption='Binary Mask', use_container_width=True)

                row1, row2, row3 = st.columns(3)
                with row1:
                    st.markdown(f"📐 **Polyp Area:** {area} px")
                with row2:
                    st.markdown(f"🔢 **Estimated Polyp Count:** {count_range}")
                with row3:
                    st.markdown(f"📊 **Coverage:** {coverage:.2f}%")

                row4, row5, row6 = st.columns(3)
                with row4:
                    st.markdown(f"🔥 **Severity:** {severity_text}")
                with row5:
                    st.markdown(f"⚠️ **Risk Level:** {risk_level}")
                with row6:
                    st.markdown(f"🧠 **Model Confidence:** {confidence_score:.2f}%")
                with row6:
                    st.empty()

                st.markdown("#### 📈 Severity Level")
                st.markdown(f'''
                <div style="background: linear-gradient(to right, lightgreen 25%, yellow 25%, orange 50%, red 75%);
                            border-radius: 10px; height: 25px; position: relative; margin-bottom: 20px;">
                    <div style="position: absolute; left: {severity_score * 33.33}%; top: -10px;
                                width: 0; height: 0; border-left: 6px solid transparent;
                                border-right: 6px solid transparent; border-bottom: 10px solid black;"></div>
                </div>
                ''', unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center; font-size:14px;'>Level: <b>{severity_text}</b></p>", unsafe_allow_html=True)

                st.markdown("<hr style='margin:30px 0;'>", unsafe_allow_html=True)

                mask_filename = f"mask_{name_wo_ext}.jpg"
                pred_pil_image.save(os.path.join(masks_dir, mask_filename), "JPEG")

                results_summary.append({
                    "Image Name": original_file_name,
                    "Estimated Polyp Count": count_range,
                    "Polyp Area": area,
                    "Image Area": image_area,
                    "% Coverage": f"{coverage:.2f}",
                    "Severity": severity_text,
                    "Risk Level": risk_level
                })

                global_index += 1

        zip_path = os.path.join(temp_dir, "masks.zip")
        with ZipFile(zip_path, 'w') as zipf:
            for mask_file in os.listdir(masks_dir):
                zipf.write(os.path.join(masks_dir, mask_file), arcname=mask_file)

        with open(zip_path, "rb") as f:
            st.download_button("Download All Masks as ZIP", data=f.read(), file_name="masks.zip", mime="application/zip")

        if results_summary:
            df_summary = pd.DataFrame(results_summary)
            st.subheader("📊 Segmentation Summary Table")
            st.dataframe(df_summary)

            csv = df_summary.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Report", data=csv, file_name="polyp_summary.csv", mime="text/csv")

        shutil.rmtree('./temp_app_data')

    except Exception as e:
        st.error(f"An error occurred while processing the images: {e}")


def main():
    st.set_page_config(page_title="Polyp Segmentation Tool", layout="wide")

    st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #ffffff;
            color: #333333;
        }
        .stButton > button {
            background-color: #1976D2;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 0.5rem 1.2rem;
        }
        .stButton > button:hover {
            background-color: #1565C0;
        }
        .stDownloadButton > button {
            background-color: #2E7D32;
            color: white;
            border-radius: 8px;
            font-size: 15px;
        }
        .stDownloadButton > button:hover {
            background-color: #1B5E20;
        }
        hr {
            border-top: 1px solid #ccc;
            margin-top: 25px;
            margin-bottom: 25px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center; color:#1565C0;'>Polyp Segmentation Tool</h1>", unsafe_allow_html=True)
    st.image("https://production-media.paperswithcode.com/datasets/Screenshot_from_2021-05-05_23-44-10.png", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style='color:#1565C0;'>📘 How to Use</h3>
    <ul style='font-size:15px; color:#444444;'>
        <li>Upload one or more colonoscopy images (JPG or JPEG)</li>
        <li>Click “Process Images” to run segmentation</li>
        <li>View visual overlays, severity, area, and polyp count</li>
        <li>Download all results as ZIP or CSV</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Colonoscopy Images", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select one or more JPG images",
        type=['jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload colonoscopy images for segmentation",
    )

    model_path = 'model/best_model_b4.h5'
    model = load_model(model_path)

    if model is None:
        st.error("🚨 Model loading failed. Please check your model file.")
        return

    temp_dir = './temp_app_data/app_images/test'

    if uploaded_files:
        st.markdown("### 🔍 Process Uploaded Images", unsafe_allow_html=True)
        process_btn = st.button("🚀 Process Images")

        if process_btn:
            with st.spinner("🧠 Running segmentation..."):
                process_and_display_images(model, uploaded_files, temp_dir)
                st.success("✅ Segmentation complete!")
    else:
        st.info("Upload images above to begin.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 13px; color: gray;'>
        Built using streamlit<br>
        Major Project 2025
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

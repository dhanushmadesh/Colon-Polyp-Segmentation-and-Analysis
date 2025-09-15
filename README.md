# 🩺 Colon Polyp Segmentation & Analysis

A Deep Learning project that detects and segments **colon polyps** from colonoscopy images using **U-Net with EfficientNetB4**.  
The tool provides **visual overlays**, **numeric analysis**, and **natural-language summaries** to assist in research and awareness.

---

## 🚀 Key Features
- **Segmentation**: Detects and highlights polyp regions in colonoscopy images.  
- **Analysis**: Reports polyp count, area, shape (solidity, eccentricity, aspect ratio), and coverage %.  
- **Summaries**: Generates descriptive text for each image (e.g., *“medium-sized oval polyp with smooth borders”*).  
- **Outputs**:
  - Overlay image (original + mask)  
  - Numeric CSV report  
  - Downloadable masks (ZIP)  

---

## 📊 Results
- **F1-score**: 0.886  
- **IoU**: 0.796  
- **PR AUC**: 0.898  
- **Best Model**: `model/best_model_b4.h5`

![Precision-Recall Curve](precision_recall_curve.png)

---

## 🎯 Demo
1. Upload colonoscopy images  
2. Click **Run Segmentation**  
3. View results:
   - Original | Overlay | Mask  
   - Numeric table  
   - Summary description  
   - CSV + ZIP downloads  

---

## 👨‍💻 Team
- Dhanush  
- Hemanth  
- Sathya  
- Likhith  

---

⚠️ *This project is for academic and research purposes only — not a clinical diagnostic tool.*




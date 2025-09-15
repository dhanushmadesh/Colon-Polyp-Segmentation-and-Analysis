# 🩺 Colon Polyp Segmentation & Analysis

> **Major Project (2025) – Information Science & Engineering**  
> Deep Learning–based system for detecting and segmenting colon polyps in colonoscopy images.  

---

## 🚀 Overview

Colorectal cancer is one of the most common cancers worldwide. Early detection of **polyps** during colonoscopy plays a vital role in prevention.  
This project builds a **Deep Learning pipeline** using **EfficientNetB4 + UNet** to automatically:

- Detect polyps in colonoscopy images.  
- Generate segmentation masks.  
- Highlight polyps with overlay visualization.  
- Provide **scientific analysis metrics**. 
- Export results as **CSV report** and **segmentation masks (ZIP)**.  

---

## ✨ Features

- 🧠 **Deep Learning Model**: EfficientNetB4 backbone with UNet architecture.  
- 📸 **Image Segmentation**: Produces binary masks of detected polyps.  
- 🎨 **Overlay Visualization**: Polyps highlighted on the original image.  
- 📊 **Analysis Table**:  
  - Polyp area (px)  
  - Estimated polyp count  
  - Coverage %  
  - Severity (small/medium/large)  
  - Risk level (low/moderate/high)  
  - Confidence score (%)  
- 💾 **Export Options**:  
  - CSV summary report  
  - ZIP of all mask images  
- 🌐 **Interactive UI**: Built with **Gradio** (no need for complex setup).  

---

## 🖼️ Demo UI

[ Upload Images ] → [ Run Segmentation ]
↓
[ Original | Overlay | Mask ] results in grid
↓
📊 Summary Table
↓
⬇️ Download CSV | ZIP



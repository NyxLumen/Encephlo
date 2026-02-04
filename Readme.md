# Encephlo | Neuro-Oncology Triage System

**A Hybrid Ensemble Framework for Interpretable Brain Tumor Detection**

<p align="center">
<img src="https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python" alt="Python"/>
<img src="https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow" alt="Tensorflow"/>
<img src="https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit"/>
</p>

## Overview

**Encephlo** is an end-to-end Clinical Decision Support System (CDSS) designed to bridge the gap between "Black Box" AI and clinical interpretability.

Unlike standard classifiers that rely on a single model, Encephlo utilizes a **Heterogeneous Ensemble (The Council of Experts)** combining three distinct neural architectures to maximize robustness. It features a fully integrated clinical workflow, supporting raw medical data (**DICOM**), Explainable AI (**Grad-CAM**), and automated patient reporting.

## Key Features

- **🧠 Hybrid Ensemble Engine:** A weighted consensus system using **EfficientNetB0** (Efficiency), **MobileNetV2** (Speed), and **DenseNet121** (Texture Analysis) to minimize false negatives.

- **👁️ Explainable AI (XAI):** Real-time **Grad-CAM** visualization overlays heatmaps on MRI scans, allowing doctors to verify if the model is looking at the tumor or artifacts.

- **🏥 Medical Standard Support:(WIP)** Native support for **DICOM (`.dcm`)** files, extracting patient metadata alongside image data.

- **⚙️ Smart Preprocessing:** Automated **Otsu’s Thresholding** pipeline to strip skull and background noise from scans before analysis.

- **📄 Automated Reporting:** Generates instant, downloadable **PDF Clinical Reports** with natural-language diagnostic recommendations.

## Tech Stack

- **Deep Learning:** TensorFlow, Keras (Ensemble Architecture)
- **Web Interface:** Streamlit (Custom CSS for Clinical UI)
- **Image Processing:** OpenCV (Otsu/Contour), Pillow
- **Medical Data:** Pydicom (DICOM handling)
- **Reporting:** FPDF (PDF Generation)
- **Analytics:** Scikit-learn, Matplotlib, Seaborn

## 📂 Project Structure

```bash
/encephlo
├── /src
│   ├── app.py           # Main Clinical Dashboard Application
│   ├── utils.py         # Image Preprocessing & Grad-CAM Logic
│   ├── report.py        # PDF Generation Engine
│   └── evaluate.py      # Validation Scripts (Confusion Matrix/Metrics)
├── /models              # Saved Ensemble Weights (.h5)
├── /notebooks           # Training pipelines for EfficientNet/MobileNet/DenseNet
└── requirements.txt     # Dependency list
```

## The Team

NyxLumen – System Architect & Lead Developer

    Designed the full-stack architecture, developed the Streamlit clinical interface, implemented the Otsu preprocessing pipeline, and engineered the PDF reporting module.

sid-gupta-007 – Lead ML Engineer

    Designed the Neural Network architecture, trained the EfficientNet/MobileNet/DenseNet models, and implemented the Soft Voting ensemble logic.

AbelBobby – QA Engineer & Data Operations

    Managed dataset curation and isolation (Train/Test splits), performed Black Box system testing, and conducted the comparative performance analysis (Ablation Studies).

## ⚠️ Disclaimer

This system is a prototype for educational and research purposes only. It is not FDA-approved and should not be used for actual medical diagnosis without clinical validation.

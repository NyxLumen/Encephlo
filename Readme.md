<h1 align="center">Encephlo</h1>
<p align="center">
  <b>Tri-Model Feature Fusion Architecture for Neuro-Oncological Classification</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-99.47%25-success?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/False_Negatives-0-success?style=for-the-badge" alt="False Negatives">
  <img src="https://img.shields.io/badge/Architecture-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch">
</p>

<p align="center">
  <img src="research/xai_4class_grid.png" alt="XAI Thermal Overlays">
</p>

The deployment of deep learning in neuro-oncology is frequently bottlenecked by "shortcut learning"—where standard neural networks memorize background artifacts (like skull boundaries or watermarks) rather than analyzing actual biological pathology.

Encephlo is an architectural solution to this dataset bias. By decapitating standard classification layers and mathematically fusing dense local textures, spatial gradients, and global structural dependencies, this pipeline achieves **99.47% accuracy** with **zero false negatives** across four biological classes (Glioma, Meningioma, Pituitary Adenoma, and Healthy Tissue).

## The Diagnostic Pipeline

The Encephlo architecture is structured into three mathematically rigid phases to ensure absolute diagnostic stability:

> **1. Anatomical Pre-Processing (OpenCV)** > To definitively prevent shortcut learning prior to inference, an aggressive OpenCV contouring algorithm auto-crops the MRI, completely masking the skull and non-anatomical voids. Contrast Limited Adaptive Histogram Equalization (CLAHE) is then applied to hyper-pronounce soft tissue boundaries.

> **2. Tri-Model Feature Extraction** > The cleaned tensors are processed in parallel through three orthogonal computer vision paradigms. The final classification layers are removed to output raw mathematical feature arrays.

<table align="center" width="100%">
  <tr>
    <td align="center" width="33%"><b>DenseNet121</b><br><code>1024-D Vector</code></td>
    <td align="center" width="33%"><b>EfficientNet-B0</b><br><code>1280-D Vector</code></td>
    <td align="center" width="33%"><b>ViT-B/16</b><br><code>768-D Vector</code></td>
  </tr>
  <tr>
    <td valign="top">Isolates dense cellular textures and localized micro-anomalies.</td>
    <td valign="top">Captures multi-scale spatial gradients and macroscopic tumor edges.</td>
    <td valign="top">Vision Transformer utilizing self-attention to map long-range structural dependencies and global anatomical geometry.</td>
  </tr>
</table>

> **3. Master SVM Fusion** > The three arrays are horizontally concatenated into a singular, highly dense **3072-Dimensional feature vector**. Rather than relying on softmax probabilities, this tensor is evaluated by a Support Vector Machine (SVM) utilizing a Radial Basis Function (RBF) kernel, translating deep learning features into rigid mathematical decision boundaries.

## Mathematical Separation & Performance

In a clinical setting, a false negative is the most catastrophic failure mode. The Tri-Model SVM Fusion completely eliminates the spatial blind spots of the isolated models, achieving a perfect 405/405 classification rate for healthy tissue (0 false negatives).

<p align="center">
  <img src="research/SVM_Fusion_Confusion_Matrix.png" alt="Master SVM Confusion Matrix">
</p>

To visually prove the pipeline is not executing probabilistic guesswork, t-SNE dimensionality reduction was applied to the 3072-D extracted feature space. The vectors naturally gravitate into four strictly isolated topological islands, proving the deep learning models learned defining biological geometries.

<p align="center">
  <img src="research/SVM_Fusion_tSNE_Plot.png" width="800" alt="t-SNE Visualization">
</p>

## 🔬 Deep Dive: Isolated Model Metrics

<details>
<summary><b>Click to expand: Isolated Model Error Distributions & Training Dynamics</b></summary>
<br>

To mathematically justify the computational overhead of the 3072-D fusion, we benchmarked the isolated paradigms prior to concatenation. The orthogonal error distributions below prove exactly why a single-model approach is clinically insufficient.

<p align="center">
  <img src="research/DenseNet121_confusion_matrix.png" width="32%" alt="DenseNet121 Matrix">
  <img src="research/EfficientNet_B0_confusion_matrix.png" width="32%" alt="EfficientNet-B0 Matrix">
  <img src="research/ViT_B_16_confusion_matrix.png" width="32%" alt="ViT-B/16 Matrix">
</p>

> **The Orthogonal Flaw:** Notice how ViT-B/16 _(right)_ misclassifies 10 Gliomas due to diffuse boundaries, while EfficientNet-B0 _(center)_ successfully resolves those exact spatial gradients. This proves the models fail in different regions of the feature space, making them perfect candidates for fusion.

<br>

<p align="center">
  <img src="research/vit_b_16_training_graphs.png" width="600" alt="ViT-B/16 Training Dynamics">
</p>

> **Transformer Convergence:** Training and validation curves for ViT-B/16. The erratic spikes in the early epochs represent the characteristic volatility of self-attention mechanisms actively mapping global structural geometry before stabilizing.

</details>

## Core Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
</p>

- **Deep Learning Framework:** PyTorch & Torchvision _(Model architecture, decapitation, and gradient hooking)_
- **Classical Machine Learning:** Scikit-Learn _(Master SVM with RBF kernel, t-SNE dimensionality reduction)_
- **Computer Vision:** OpenCV & Pillow _(Anatomical auto-cropping, CLAHE contrast enhancement, thermal heatmap blending)_
- **Scientific Computing:** NumPy _(High-dimensional vector concatenation and tensor normalization)_

## Contributors

- **Aditya Sharma** ([@NyxLumen](https://github.com/NyxLumen))
- **Siddharth Gupta** ([@sid-gupta-007](https://github.com/sid-gupta-007))
- **Abel Bobby** ([@AbelBobby](https://github.com/AbelBobby))

<p align="center">
    Made with love ❤️
</p>

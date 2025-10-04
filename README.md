
This repository contains solutions for the two tasks assigned in the project:

- **Q1:** Vision Transformer (ViT) on CIFAR-10
- **Q2:** Text-Driven Image Segmentation using SAM 2

## Q1 — Vision Transformer on CIFAR-10

**Goal:** Implement a ViT model and train on CIFAR-10 (10 classes) to achieve high test accuracy.

**Implementation Highlights:**
- Images are patchified (configurable patch size).
- Learnable positional embeddings added.
- CLS token prepended for classification.
- Transformer encoder blocks: Multi-Head Self-Attention + MLP + residual connections + normalization.
- Classification performed using the CLS token embedding.
- Trained using PyTorch, with optional data augmentations.


**Results:**
| Dataset | Test Accuracy |
|---------|---------------|
| CIFAR-10 | 72.95% |



## Q2 — Text-Driven Image Segmentation with SAM 2

**Goal:** Segment a chosen object in an image using a text prompt.

**Pipeline:**
1. Load the input image.
2. Accept a text prompt (e.g., `"car"`, `"person"`).
3. Convert text prompt to region proposals using **GroundingDINO** (or similar model).
4. Feed proposals to **SAM 2** for pixel-level segmentation.
5. Display segmentation mask overlay on original image.

**Limitations:**
- Works best for clear objects; overlapping or small objects may be less accurate.
- Pretrained models are not included; must be downloaded separately (weights can be fetched at runtime in Colab).

**How to run in Colab:**
1. Open `q2.ipynb` in Google Colab.
2. Run the first cell to install dependencies.
3. Upload your image and run all cells sequentially.
4. Enter a text prompt when prompted.

---

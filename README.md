
This repository contains solutions for the two tasks assigned in the project:

- **Q1:** Vision Transformer (ViT) on CIFAR-10
- **Q2:** Text-Driven Image Segmentation using SAM 2

## Q1 — Vision Transformer on CIFAR-10

**Goal:** Implement a ViT model and train on CIFAR-10 to achieve high test accuracy.

**Implementation Highlights:**
- Images are patchified.
- Learnable positional embeddings added.
- CLS token prepended for classification.
- Transformer encoder blocks: Multi-Head Self-Attention + MLP + residual connections + normalization.
- Classification performed using the CLS token embedding.
- Trained using PyTorch, with data augmentations.
**Configuration:**
- optimizer = "lr_scheduler"
- learning_rate = 3
- batch_size = 128
- epochs = 150
- loss_function = "CrossEntropyLoss"
**How to run in Colab:**
1. Open `q1.ipynb` in Google Colab.
2. Ensure runtime is set to GPU under Runtime > Change runtime type > GPU.
3. Run the first cells to install dependencies.
4. Run all cells sequentially
**Results:**
| Dataset | Test Accuracy |
|---------|---------------|
| CIFAR-10 | 72.95% |



## Q2 — Text-Driven Image Segmentation with SAM 2

**Goal:** Segment a chosen object in an image using a text prompt.

**Pipeline:**
1. Load the input image.
2. Accept a text prompt (e.g., `"car"`, `"person"`).
3. Convert text prompt to region proposals.
4. Feed proposals to **SAM 2** for pixel-level segmentation.
5. Display segmentation mask overlay on original image.

**Limitations:**
- Works best for clear objects; overlapping or small objects may be less accurate.
- Pretrained models are not included; must be downloaded separately (weights can be fetched at runtime in Colab).

**How to run in Colab:**
1. Open `q2.ipynb` in Google Colab.
2. Run the first cells to install dependencies.
3. Provide image path and run all cells sequentially.
4. Enter a text prompt when prompted for object.

---

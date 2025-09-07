# 🧠 MNIST Digit Classification with CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch. The architecture is compact, optimized for learning from 28x28 grayscale images, and avoids fully connected layers in favor of a fully convolutional approach with Global Average Pooling.

---

## 📊 Dataset: MNIST

- **Images**: 28x28 grayscale
- **Classes**: 10 digits (0–9)
- **Channels**: 1
- **Type**: Classification

---

## 🏗️ Model Architecture Overview

```
Input (1x28x28)
│
├── Conv2d(1, 8, 3x3) → ReLU
├── Conv2d(8, 8, 3x3) → ReLU → MaxPool(2x2)
├── Conv2d(8, 16, 3x3) → ReLU
├── Conv2d(16, 16, 3x3) → ReLU → MaxPool(2x2)
├── Conv2d(16, 32, 3x3) → ReLU
├── Conv2d(32, 10, 1x1)
├── AdaptiveAvgPool2d(1x1)
└── LogSoftmax
```


---

## 🔍 Layer-by-Layer Breakdown

### 🔹 Layer 1: `Conv2d(1, 8, kernel_size=3, stride=1, bias=False)` + ReLU

- **Purpose**: First layer to learn low-level features like edges.
- **Reasoning**:
  - Kernel size 3x3: Good for local pattern extraction.
  - No bias: BatchNorm is not used here, but removing bias can simplify training slightly.
  - 8 filters: Small number, enough for low-level features, keeps model lightweight.

### 🔹 Layer 2: `Conv2d(8, 8, 3x3)` + ReLU + `MaxPool2d(2x2)`

- **Purpose**: Refine features from layer 1 and reduce spatial size.
- **MaxPooling**: Reduces feature map from 26x26 → 13x13.
- **Reasoning**:
  - Keeps number of channels the same.
  - Pooling helps make the model spatially invariant and reduces computation.

### 🔹 Layer 3: `Conv2d(8, 16, 3x3)` + ReLU

- **Purpose**: Start learning more abstract patterns.
- **Reasoning**:
  - Doubles number of filters to capture more complex patterns.

### 🔹 Layer 4: `Conv2d(16, 16, 3x3)` + ReLU + `MaxPool2d(2x2)`

- **Purpose**: Further abstraction and downsampling.
- **Effect**: 11x11 → 5x5 after pooling.
- **Reasoning**:
  - More filters maintained to continue capturing richer features.
  - Pooling again helps reduce overfitting and size.

### 🔹 Layer 5: `Conv2d(16, 32, 3x3)` + ReLU

- **Purpose**: Deepest features, more semantic abstraction.
- **Effect**: Input 5x5 → Output 3x3.
- **Reasoning**:
  - Increases filters again for final feature refinement.

### 🔹 Layer 6: `Conv2d(32, 10, 1x1)`

- **Purpose**: Reduce feature maps to match number of classes (10).
- **Reasoning**:
  - 1x1 convolution allows channel-wise transformation without affecting spatial dimensions.
  - Converts each of the 3x3 outputs into a 10-channel feature map (still 3x3 spatially).

### 🔹 Global Average Pooling: `AdaptiveAvgPool2d(1x1)`

- **Purpose**: Aggregate spatial information into a single score per class.
- **Reasoning**:
  - Replaces fully connected layers.
  - Reduces risk of overfitting.
  - Fully convolutional: can adapt to varying input sizes.

### 🔹 Output: `log_softmax(x, dim=1)`

- **Purpose**: Converts class scores to log-probabilities for training with `NLLLoss`.
- **Reasoning**:
  - Log-Softmax is numerically stable and pairs well with negative log-likelihood loss.

---

## 📐 Tensor Size Transformations

| Layer          | Output Shape         | Description                       |
|----------------|----------------------|-----------------------------------|
| Input          | (1, 28, 28)          | Grayscale input                   |
| Conv1          | (8, 26, 26)          | 3x3 conv, no padding              |
| Conv2 + Pool   | (8, 12, 12)          | Downsampled by MaxPool           |
| Conv3          | (16, 10, 10)         | 3x3 conv                          |
| Conv4 + Pool   | (16, 4, 4)           | Downsampled by MaxPool           |
| Conv5          | (32, 2, 2)           | 3x3 conv                          |
| Conv6 (1x1)    | (10, 2, 2)           | 10 output channels (classes)     |
| GAP            | (10, 1, 1)           | Global average pooling           |
| Flatten        | (10,)                | Final logits per class           |

---

## ✅ Advantages of This Architecture

- **Fully Convolutional**: No dense layers, fewer parameters.
- **Lightweight**: Suitable for training on CPU or small GPU.
- **Well-Regularized**: Uses max pooling and GAP instead of FC layers.
- **Flexible**: GAP enables adapting to other image sizes with minimal changes.

---

## 📦 Possible Improvements

- Add **BatchNorm** after conv layers for better training stability.
- Add **Dropout** for regularization.
- Use **padding=1** in conv layers to preserve spatial dimensions.
- Add **data augmentation** to improve generalization.


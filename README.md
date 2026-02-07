# Project Overview

## Project Goal
This project explores practical applications of deep learning and gradient descent in two distinct areas:
- Generating **"thread art"** using optimization.
- Demonstrating the power of **neural network embeddings** for improved image classification.

It highlights how complex data can be made linearly separable through learned feature representations.

---

## What We Did and Results

### Problem 1: Thread Art Generation

**Objective:**  
Recreate target images by optimizing the intensity of “threads” connecting pegs on a circular loom, framed as a gradient descent problem.

**Methodology:**  
Implemented a PyTorch-based solution where thread configurations were converted to pixel intensities. The Mean Squared Error (MSE) loss between the generated and target image was minimized using the **Adam optimizer**.

**Results:**  
Successfully generated thread art representations of various target images (e.g., a star) with a low final MSE loss of approximately **0.027458** after **1000 iterations**, showcasing the effectiveness of gradient-based optimization for creative tasks.

---

### Problem 2: Cats vs. Dogs Classification using Deep Embeddings

**Objective:**  
Demonstrate that animal classes (cats and dogs) that are **not linearly separable** in raw pixel space become **nearly linearly separable** when represented by features extracted from a pre-trained deep neural network.

#### Part 1 - Raw Pixels
**Methodology:**  
Trained a simple linear classifier on raw **CIFAR-10** pixel data (3072 dimensions) for cats and dogs.

**Results:**  
Achieved a baseline training accuracy of **0.6303** with a final loss of **0.678981**, confirming the non-separability of these classes in raw pixel space.

#### Parts 2 & 3 - ResNet-18 Embeddings
**Methodology:**  
Extracted **512-dimensional features** from cat and dog images using the penultimate layer of a **pre-trained ResNet-18** model. Then trained another simple linear classifier on these embeddings.

**Results:**  
Achieved a significantly higher training accuracy of **0.8970** and a much lower loss of **0.238778**, demonstrating a **0.2667 improvement** in accuracy.  
This highlights the superior representational power of deep network embeddings.

#### Part 4 - 2D Visualization
**Methodology:**  
Applied **Principal Component Analysis (PCA)** to reduce the 512-dimensional embeddings to **2 dimensions** for visual inspection.

**Results:**  
The 2D PCA plot clearly showed distinct clusters for cats and dogs, visually confirming the **near linear separability** of these classes in the embedded feature space.

---

## Technologies Used

- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`, `torchvision`)  
- **Data Manipulation & Analysis:** NumPy (`numpy`), `os`, `zipfile`, PIL (`Pillow`)  
- **Machine Learning:** Scikit-learn (`sklearn.decomposition.PCA`)  
- **Data Visualization:** Matplotlib (`matplotlib.pyplot`)  
- **Utilities:** `gdown` (for file downloads)  
- **Hardware Acceleration:** GPU (CUDA) support for PyTorch operations

# Digital-Image-Processing
Assignments for Digital-Image-Processing course at ECE AUTH

# Digital Image Processing – Assignment 1  
Histogram Equalization & Matching 

## Overview

In this project, I implemented histogram processing techniques for grayscale images in Python.

Specifically, I implemented:

- Histogram Equalization
- Histogram Matching

Each method was developed using three different modes:

- Greedy
- Non-Greedy
- Post-Disturbance (noise-based approach)

## Files

- `hist_utils.py` – Histogram computation and transformation utilities  
- `hist_modif.py` – Core implementation of equalization and matching algorithms  
- `demo.py` – Demonstration script that loads images, runs all modes, and displays results  
- `report.pdf` – Analysis and experimental results  

## Usage

Run:

```bash
python demo.py
```

The program asks for:
- Input image path
- Reference image path

It then performs histogram equalization and histogram matching in all three modes and displays the results with their corresponding histograms.

---

This project demonstrates the implementation and comparison of different histogram modification strategies and their effect on image contrast and distribution.

---

# Digital Image Processing – Assignment 2 
Edge & Circle Detection 

## Overview

In this project, I implemented edge detection and circle detection algorithms from scratch using FIR convolution.

Specifically, I implemented:

- 2D FIR convolution
- Sobel edge detection (with threshold analysis)
- Laplacian of Gaussian (LoG) edge detection
- Circular Hough Transform for circle detection

## Implemented Methods

- **Sobel Operator**  
  Computes first-order gradients (Gx, Gy) and detects edges via thresholding on gradient magnitude.

- **Laplacian of Gaussian (LoG)**  
  Applies Gaussian smoothing and Laplacian filtering, detecting edges through zero-crossings.

- **Circular Hough Transform**  
  Detects circles in binary edge images using a 3D voting accumulator (center_x, center_y, radius).

## Demo

The `demo.py` script:

- Analyzes Sobel for multiple threshold values
- Plots threshold vs detected edge points
- Compares Sobel and LoG edge maps
- Detects circles using:
  - Sobel + Hough
  - LoG + Hough
- Evaluates different voting thresholds (V_min)

---

This project demonstrates implementation of convolution-based edge detection and parametric shape detection using the Hough Transform.

---

# Digital Image Processing – Assignment 3  
Image Segmentation  

## Overview

In this project, I implemented graph-based image segmentation methods.

Specifically, I implemented:

- Image-to-graph representation (fully connected affinity graph)
- Spectral Clustering for image segmentation
- Normalized Cuts (n-cuts)
  - Non-recursive version
  - Recursive version with stopping criteria

## Implemented Methods

- **Image as Graph**
  - Each pixel is treated as a node
  - Edge weights are defined as:
    A(i, j) = 1 / exp(d(i, j))
  - d(i, j) is the Euclidean distance between pixel color vectors
  - Produces a fully connected affinity matrix

- **Spectral Clustering**
  - Laplacian: L = D − W
  - Solve eigenproblem: Lx = λx
  - Use k smallest eigenvectors
  - Apply KMeans (random_state=1) in spectral space

- **Normalized Cuts (n-cuts)**
  - Solve generalized eigenproblem: Lx = λDx
  - Cluster using KMeans on eigenvectors
  - Compute Ncut metric to evaluate partition quality

- **Recursive n-cuts**
  - Repeatedly splits graph into 2 clusters
  - Stops when:
    - Cluster size < T1
    - Ncut value > T2
  - Produces hierarchical segmentation

## Demo

The demo scripts:

- `demo1.py` – Spectral clustering on a predefined affinity matrix
- `demo2.py` – Spectral clustering on RGB images
- `demo3a.py` – Non-recursive n-cuts (k = 2,3,4)
- `demo3b.py` – Single-step binary partition with Ncut evaluation
- `demo3c.py` – Full recursive n-cuts (T1=5, T2=0.2)

---

This project demonstrates graph-based segmentation using spectral methods and highlights the differences between spectral clustering and normalized cuts, including recursive hierarchical splitting.

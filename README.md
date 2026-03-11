# Stereo Matching and Homography (DLT)

Computer vision project implementing stereo depth estimation and planar image warping using classical vision algorithms.

## Overview

This project implements two core computer vision pipelines:

1. **Stereo Matching for Depth Estimation**
2. **Homography Estimation and Image Warping using DLT**

The implementation focuses on algorithmic computer vision techniques, including stereo correspondence, disparity computation, homography estimation, and image warping.

---

# Part 1 — Stereo Matching

Depth is estimated from a pair of stereo images by computing the **disparity map** between corresponding pixels.

Two matching methods were implemented:

### SSD (Sum of Squared Differences)

For each pixel, a window is compared across a disparity range to find the minimum SSD score.

The disparity is defined as the horizontal shift that minimizes:

$$
SSD(L,R) = \sum_i (L_i - R_i)^2
$$

### Normalized Correlation

To improve robustness to illumination changes, disparity is also computed using normalized correlation.

The disparity is selected based on the maximum correlation score.

### Output

The pipeline produces a **disparity map**, which approximates the scene depth.

---

# Part 2 — Homography and Image Warping

The project also implements planar image transformation using homography.

### Homography Estimation (DLT)

The homography matrix is estimated using the **Direct Linear Transform (DLT)** method.

Steps:

1. Select 4+ matching points between two images
2. Construct the linear system
3. Solve using **Singular Value Decomposition (SVD)**
4. Normalize the homography matrix

### Image Warping

Using the estimated homography:

- the source image is projected
- the image is warped into the destination plane
- the result is blended using a mask

This enables applications such as:

- image projection
- panorama stitching
- billboard replacement

---

# Technologies

- Python
- NumPy
- OpenCV
- PyTorch

---

# Skills Demonstrated

- Stereo Correspondence
- Disparity Estimation
- Homography Estimation
- Direct Linear Transform (DLT)
- Image Warping
- Algorithmic Computer Vision
- Numerical Linear Algebra

---

# Example Results

The project generates:

- disparity maps from stereo images
- warped images using homography transformation


----

###### Ariel University, Israel || Semester B, 2021

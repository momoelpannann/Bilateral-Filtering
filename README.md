# Bilateral-Filtering

## Overview
This project implements a **manual bilateral filter** for image processing and compares it with OpenCV's `cv2.bilateralFilter`. Bilateral filtering reduces noise while preserving edges, making it useful for detailed images. The project supports both grayscale and color images.

---

## Features
- Custom bilateral filtering for noise reduction and edge preservation.
- Comparison with OpenCV's `cv2.bilateralFilter`.
- Supports grayscale and color images.
- Visualizes results side-by-side using `matplotlib`.

---

## Requirements
- Python 3.x
- Libraries:
  - `numpy`
  - `opencv-python`
  - `matplotlib`

Install dependencies:
```bash
pip install numpy opencv-python matplotlib
```
## File Structure

The project is organized as follows:
- **`bilateral_filter.py`**: The main Python script for manual bilateral filtering.
- **`README.md`**: Documentation for the project.


# How It Works

## Input
The script accepts:
- **A file path** to the input image (grayscale or color).
- **Filter parameters** entered by the user:
  - `d`: Diameter of the pixel neighborhood (e.g., 5, or 0 for OpenCV’s automatic size).
  - `sigma_range`: Standard deviation for intensity similarity.
  - `sigma_space`: Standard deviation for spatial proximity.

---

## Processing
1. **Manual Filtering**:
   - Computes Gaussian weights for spatial and intensity differences.
   - Applies these weights to the local pixel neighborhood to compute the filtered value.

2. **Color Image Handling**:
   - Converts images to Lab color space.
   - Filters each channel (`L`, `a`, `b`) separately using the manual bilateral filter.
   - Merges the filtered channels and converts them back to the BGR color space.

3. **Comparison with OpenCV**:
   - Uses OpenCV’s `cv2.bilateralFilter` with the same parameters to provide a benchmark.

---

## Output
- Displays the following results in a `matplotlib` window:
  - **Left**: Manual filtering results.
  - **Right**: OpenCV filtering results.
    
# Example Workflow

1. Run the script in your terminal or IDE.
2. Provide the path to an image file when prompted (grayscale or color).
3. Enter the desired parameters for the bilateral filter.
4. View the results in a `matplotlib` window comparing the manual and OpenCV-filtered images.

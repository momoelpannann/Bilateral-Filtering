import numpy as np
import cv2
import matplotlib.pyplot as plt

def bilateral_filter_manual(image, d, sigma_range, sigma_space):
    """
    Perform bilateral filtering manually to mimic OpenCV's behavior.

    Parameters:
        image (numpy.ndarray): Input grayscale or single-channel image.
        d (int): Diameter of the pixel neighborhood (0 for automatic size).
        sigma_range (float): Standard deviation for range Gaussian (intensity similarity).
        sigma_space (float): Standard deviation for spatial Gaussian (geometric closeness).

    Returns:
        numpy.ndarray: Bilateral filtered image.
    """
    # Convert image to float32 for precision matching with OpenCV
    image = image.astype(np.float32)

    # Determine the radius of the neighborhood based on the diameter (d)
    if d == 0:
        # Automatic size: twice the spatial sigma + 1 ensures sufficient neighborhood coverage
        radius = int(2 * sigma_space + 1)
    else:
        # Explicit diameter provided; calculate radius
        radius = d // 2

    # Pad the image to handle boundary pixels by mirroring edges
    padded_image = np.pad(image, pad_width=radius, mode='reflect')

    # Precompute the spatial Gaussian kernel based on the geometric closeness of pixels
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    # Spatial Gaussian: e^(-(x^2 + y^2) / (2 * sigma_space^2))
    spatial_gaussian = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))

    # Create an empty output image to store filtered results
    filtered_image = np.zeros_like(image)

    # Process each pixel in the input image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the local neighborhood centered at the current pixel
            region = padded_image[i:i + 2 * radius + 1, j:j + 2 * radius + 1]

            # Compute the range Gaussian based on intensity differences
            center_intensity = image[i, j]  # Intensity of the central pixel
            intensity_diff = region - center_intensity
            # Range Gaussian: e^(-(difference^2) / (2 * sigma_range^2))
            range_gaussian = np.exp(-(intensity_diff**2) / (2 * sigma_range**2))

            # Combine the spatial and range Gaussians to form the bilateral filter weights
            combined_weights = spatial_gaussian * range_gaussian

            # Normalize the weights so they sum to 1
            combined_weights /= combined_weights.sum()

            # Apply the weights to the neighborhood and compute the filtered intensity
            filtered_intensity = np.sum(region * combined_weights)

            # Assign the computed value to the corresponding pixel in the output image
            filtered_image[i, j] = filtered_intensity

    # Clip the values to the valid intensity range (0 to 255) and convert back to uint8
    return np.clip(filtered_image, 0, 255).astype(np.uint8)

def main():
    # Load the image from the given file path
    image_path = input("Enter the path to the image file: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read the image. Check the file path.")
        return

    # Check if the image is grayscale or color
    is_grayscale = len(image.shape) == 2

    # User input for filter parameters
    d = int(input("Enter the diameter of the pixel neighborhood (odd number, e.g., 5 or 0 for OpenCV's automatic size): "))
    sigma_range = float(input("Enter the range standard deviation (e.g., 75.0): "))
    sigma_space = float(input("Enter the spatial standard deviation (e.g., 75.0): "))

    if is_grayscale:
        # Apply bilateral filter for grayscale images
        filtered_image_manual = bilateral_filter_manual(image, d, sigma_range, sigma_space)
        filtered_image_opencv = cv2.bilateralFilter(image, d, sigma_range, sigma_space)
    else:
        # Convert the color image to Lab color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply bilateral filtering separately to each channel
        filtered_l = bilateral_filter_manual(l_channel, d, sigma_range, sigma_space)
        filtered_a = bilateral_filter_manual(a_channel, d, sigma_range, sigma_space)
        filtered_b = bilateral_filter_manual(b_channel, d, sigma_range, sigma_space)

        # Merge the filtered Lab channels
        filtered_lab_manual = cv2.merge((filtered_l, filtered_a, filtered_b))

        # Convert back to BGR color space
        filtered_image_manual = cv2.cvtColor(filtered_lab_manual, cv2.COLOR_Lab2BGR)

        # Use OpenCV's bilateral filter for comparison
        filtered_l_opencv = cv2.bilateralFilter(l_channel, d, sigma_range, sigma_space)
        filtered_a_opencv = cv2.bilateralFilter(a_channel, d, sigma_range, sigma_space)
        filtered_b_opencv = cv2.bilateralFilter(b_channel, d, sigma_range, sigma_space)

        # Merge the OpenCV-filtered Lab channels and convert to BGR
        filtered_lab_opencv = cv2.merge((filtered_l_opencv, filtered_a_opencv, filtered_b_opencv))
        filtered_image_opencv = cv2.cvtColor(filtered_lab_opencv, cv2.COLOR_Lab2BGR)

    # Display results
    plt.figure(figsize=(10, 5))

    # Manually filtered image
    plt.subplot(1, 2, 1)
    plt.title("Manual Bilateral Filter")
    if is_grayscale:
        plt.imshow(filtered_image_manual, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(filtered_image_manual, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # OpenCV filtered image
    plt.subplot(1, 2, 2)
    plt.title("OpenCV Bilateral Filter")
    if is_grayscale:
        plt.imshow(filtered_image_opencv, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(filtered_image_opencv, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
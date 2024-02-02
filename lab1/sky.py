import cv2
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt

def apply_skyline_detection(mask, median_filter_size, height_threshold):
    """
    Detects and refines the skyline in the mask based on median filtering.

    Parameters:
        mask (numpy.ndarray): The binary mask where sky is marked with 1s and non-sky with 0s.
        median_filter_size (int): Size of the kernel used for median filtering, to smooth out the mask.
        height_threshold (int): Vertical threshold to determine where the skyline should start.
    
    Returns:
        numpy.ndarray: The refined binary mask with a clearer separation of sky.
    """
    # Iterate through each column in the mask to process it vertically.
    for column in range(mask.shape[1]):
        # Extract the vertical slice of the mask.
        column_values = mask[:, column]
        
        # Apply median filter to the column to smooth out the transitions.
        filtered_column = medfilt(column_values, median_filter_size)
        
        # Identify the transition points from sky to non-sky.
        sky_indices = np.where(filtered_column == 1)[0]
        ground_indices = np.where(filtered_column == 0)[0]
        
        # Refine the mask by filling the region between the first sky and ground indices.
        if ground_indices.size > 0 and sky_indices.size > 0:
            first_ground_index = ground_indices[0]
            first_sky_index = sky_indices[0]
            
            # Apply the height threshold to determine the skyline.
            if first_ground_index > height_threshold:
                mask[first_sky_index:first_ground_index, column] = 1
                mask[first_ground_index:, column] = 0
                mask[:first_sky_index, column] = 0
                
    return mask

def get_sky_region(img, blur_kernel_size, edge_threshold, morph_shape, median_filter_size, skyline_filter_size, skyline_threshold):
    """
    Processes the input image to detect and isolate the sky region.

    Parameters:
        img (numpy.ndarray): The input image in BGR color space.
        blur_kernel_size (tuple): The size of the kernel used for Gaussian blur.
        edge_threshold (int): The threshold used for identifying edges in the Laplacian edge detection.
        morph_shape (tuple): The size of the kernel used for morphological operations.
        median_filter_size (int): The size of the kernel used for median filtering.
        skyline_filter_size (int): The size of the kernel used for median filtering in skyline detection.
        skyline_threshold (int): The vertical threshold for skyline detection.

    Returns:
        numpy.ndarray: The part of the image identified as the sky region.
    """
    # Convert to grayscale and apply blur and median filter to smooth the image and reduce noise.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.blur(img_gray, blur_kernel_size)
    img_blurred = cv2.medianBlur(img_blurred, median_filter_size)

    # Apply Laplacian operator to detect edges and create a binary edge mask.
    laplacian_edges = cv2.Laplacian(img_blurred, cv2.CV_8U)
    edges_mask = (laplacian_edges < edge_threshold).astype(np.uint8)

    # Refine the edges mask using morphological erosion to remove small artifacts.
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_shape)
    morphed_mask = cv2.morphologyEx(edges_mask, cv2.MORPH_ERODE, morph_kernel)

    # Apply skyline detection to further refine the edges mask and isolate the sky.
    skyline_mask = apply_skyline_detection(morphed_mask, skyline_filter_size, skyline_threshold)
    
    # Mask the original image to extract the region identified as the sky.
    sky_region = cv2.bitwise_and(img, img, mask=skyline_mask)

    return sky_region

# Define the variable parameters for easy adjustment.
params = {
    'blur_kernel_size': (3, 3),      # Kernel size for Gaussian blur to smooth the image.
    'edge_threshold': 5,             # Threshold for Laplacian edge detection.
    'morph_shape': (4, 4),           # Kernel shape for morphological operations.
    'median_filter_size': 5,         # Kernel size for median filter to reduce noise.
    'skyline_filter_size': 19,       # Kernel size for median filter in skyline detection.
    'skyline_threshold': 20          # Vertical threshold for detecting the start of the sky.
}


# Function to load an image from disk
def load_image_from_disk(filepath):
    """
    Load an image from disk using OpenCV

    Parameters:
    - filepath (str): Path to the image file.

    Returns:
    - numpy.ndarray: The loaded image in BGR color space.
    """
    cv_image = cv2.imread(filepath)
    if cv_image is None:
        sys.exit("Could not read the image.")
    return cv_image

# # Test the functions with a dataset image
# if __name__ == "__main__":

#     filepath = './Dataset/image1.jpg'
#     cv_image = load_image_from_disk(filepath)

#     sky_region = get_sky_region(cv_image, **params)

#     plt.figure(figsize=(10, 10))

#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")

#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(sky_region, cv2.COLOR_BGR2RGB))
#     plt.title("Sky Region")

#     plt.show()
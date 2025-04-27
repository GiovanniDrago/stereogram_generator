import cv2
import numpy as np

def create_depth_mask(image_path):
    """
    Create a depth mask from the given image.
    
    Args:
    - image_path: str, path to the image file.
    
    Returns:
    - depth_map: np.ndarray, the generated depth map (grayscale mask).
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (Canny)
    edges = cv2.Canny(gray_image, 100, 200)

    # Invert the edges and apply a Gaussian blur to simulate depth
    depth_map = cv2.bitwise_not(edges)
    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
    
    return depth_map

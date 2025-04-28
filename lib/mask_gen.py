import cv2
import numpy as np

def create_depth_mask(image_path):
    """
    Create a smooth depth map from a colorized image by converting the image
    to grayscale and using pixel intensity to simulate depth.

    Args:
    - image_path: str, path to the input image.

    Returns:
    - depth_map: np.ndarray, the generated depth map (grayscale).
    """
    # Load the image (in color, we will convert to grayscale later)
    img = cv2.imread(image_path)

    # Convert the color image to grayscale (brightness represents depth)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image and create a smoother depth map
    blurred_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Normalize the image to create a smooth transition in depth (0 to 255)
    depth_map = cv2.normalize(blurred_gray, None, 0, 255, cv2.NORM_MINMAX)

    return depth_map

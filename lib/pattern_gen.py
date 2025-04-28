import numpy as np
from PIL import Image

def generate_pattern(width, height):
    """
    Generate a random noise pattern.

    Args:
    - width: int, width of the generated pattern.
    - height: int, height of the generated pattern.
    
    Returns:
    - pattern: np.ndarray, the generated pattern.
    """
    # Generate random noise
    noise = np.random.rand(height, width) * 255  # Random grayscale values between 0 and 255
    
    # Convert to an 8-bit grayscale image
    pattern = Image.fromarray(noise.astype(np.uint8))
    
    return pattern

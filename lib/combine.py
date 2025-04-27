import numpy as np
from PIL import Image

def create_stereogram(depth_map, pattern):
    """
    Create a stereogram using a depth map and pattern.
    Args:
    - depth_map: PIL.Image, a grayscale depth map.
    - pattern: PIL.Image, a grayscale pattern image.
    Returns:
    - stereogram: PIL.Image, the generated stereogram.
    """
    # Convert the pattern to a numpy array
    pattern_array = np.array(pattern)
    
    # Ensure the depth map is a 2D array
    depth_map_array = np.array(depth_map)
    
    # If depth map is RGB or RGBA, convert it to grayscale
    if len(depth_map_array.shape) > 2:
        # Simple conversion to grayscale by averaging channels
        depth_map_array = np.mean(depth_map_array, axis=2).astype(np.uint8)
    
    height, width = depth_map_array.shape
    
    # Make sure the pattern has the same height as the depth map
    if pattern_array.shape[0] != height:
        # Resize pattern to match depth map height
        pattern = pattern.resize((pattern_array.shape[1], height))
        pattern_array = np.array(pattern)
    
    # Initialize the stereogram array with the same shape and type as pattern_array
    stereogram = np.zeros_like(pattern_array)
    
    # Get the maximum value in the depth map for normalization
    max_depth = np.max(depth_map_array)
    
    # Loop through each row of the depth map and apply horizontal shifts based on depth values
    for y in range(height):
        for x in range(width):
            # Get the depth value for the current pixel (the shift amount)
            # Normalize the shift to a reasonable range (0-30 pixels is usually good)
            shift_amount = int((depth_map_array[y, x] / max_depth) * 30) if max_depth > 0 else 0
            
            # Compute the new x-coordinate for the shifted pixel
            new_x = (x + shift_amount) % width  # Wrap around the pattern width using modulo
            
            # Handle different image types (grayscale, RGB, RGBA)
            if len(pattern_array.shape) == 2:  # Grayscale
                stereogram[y, x] = pattern_array[y, new_x]
            else:  # RGB or RGBA
                stereogram[y, x] = pattern_array[y, new_x]
    
    # Convert the stereogram array back to a PIL image
    # Use the correct mode based on the number of channels
    if len(stereogram.shape) == 2:
        mode = 'L'  # Grayscale
    elif stereogram.shape[2] == 3:
        mode = 'RGB'
    else:
        mode = 'RGBA'
    
    stereogram_image = Image.fromarray(stereogram.astype(np.uint8), mode)
    return stereogram_image
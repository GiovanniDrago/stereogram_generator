import numpy as np
from PIL import Image

def create_stereogram(depth_map, pattern=None, depth_factor=0.1, pattern_width=None):
    """
    Create a proper autostereogram (Magic Eye) from a grayscale depth map.
    
    Args:
        depth_map: PIL Image or numpy array with grayscale depth map (white = close, black = far)
        pattern: Optional PIL Image or numpy array with the pattern to use
        depth_factor: Controls depth perception (0.05-0.2 recommended)
        pattern_width: Width of the pattern (default is 1/6 of the image width)
        
    Returns:
        PIL Image with the generated stereogram
    """
    # Convert depth map to numpy array if it's a PIL Image
    if not isinstance(depth_map, np.ndarray):
        depth_map = np.array(depth_map)
    
    # Ensure depth map is grayscale
    if len(depth_map.shape) > 2:
        depth_map = np.mean(depth_map, axis=2).astype(np.uint8)
    
    height, width = depth_map.shape
    
    # Determine pattern width
    if pattern_width is None:
        pattern_width = width // 6
    
    # Calculate max shift based on depth factor
    max_shift = int(pattern_width * depth_factor)
    
    # Create or prepare pattern
    if pattern is None:
        # Create random pattern
        is_color = np.random.choice([True, False], p=[0.3, 0.7])  # 30% chance of color
        
        if is_color:
            # RGB random pattern
            pattern = np.random.randint(0, 256, (height, pattern_width, 3), dtype=np.uint8)
            result_img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Grayscale random pattern
            pattern = np.random.randint(0, 256, (height, pattern_width), dtype=np.uint8)
            result_img = np.zeros((height, width), dtype=np.uint8)
    else:
        # Process provided pattern
        if not isinstance(pattern, np.ndarray):
            pattern = np.array(pattern)
        
        # Resize pattern if necessary
        if pattern.shape[1] != pattern_width:
            pattern_img = Image.fromarray(pattern)
            new_height = int(pattern.shape[0] * (pattern_width / pattern.shape[1]))
            pattern_img = pattern_img.resize((pattern_width, new_height))
            pattern = np.array(pattern_img)
        
        # Handle pattern height < image height by tiling vertically
        if pattern.shape[0] < height:
            repeats = (height + pattern.shape[0] - 1) // pattern.shape[0]
            if len(pattern.shape) == 3:  # Color
                tiled_pattern = np.zeros((pattern.shape[0] * repeats, pattern_width, pattern.shape[2]), dtype=pattern.dtype)
                for i in range(repeats):
                    tiled_pattern[i*pattern.shape[0]:(i+1)*pattern.shape[0], :, :] = pattern
                pattern = tiled_pattern[:height, :, :]
                result_img = np.zeros((height, width, pattern.shape[2]), dtype=np.uint8)
            else:  # Grayscale
                tiled_pattern = np.zeros((pattern.shape[0] * repeats, pattern_width), dtype=pattern.dtype)
                for i in range(repeats):
                    tiled_pattern[i*pattern.shape[0]:(i+1)*pattern.shape[0], :] = pattern
                pattern = tiled_pattern[:height, :]
                result_img = np.zeros((height, width), dtype=np.uint8)
    
    # Normalize depth map values between 0 and 1
    depth_norm = depth_map.astype(float) / 255.0
    
    # Create the stereogram using a correspondence map
    same = np.zeros(width, dtype=int)
    for y in range(height):
        # Initialize same vector for each row
        for x in range(width):
            same[x] = x
        
        # Process pixels from left to right
        for x in range(pattern_width, width):
            # Get depth-based shift for current pixel
            shift = int(max_shift * depth_norm[y, x])
            
            # Find the position to link to (with constraint handling)
            left = x - pattern_width
            right = x
            
            # Account for the depth shift
            if shift > 0:
                left += shift
                if left >= right:
                    left = right - 1
            
            # Find the link
            while same[left] != left and same[left] < right:
                left = same[left]
            
            # Establish link
            same[right] = left
            
            # Copy the pixel
            if len(result_img.shape) == 3:  # Color image
                if left < pattern_width:
                    result_img[y, x] = pattern[y % pattern.shape[0], left, :]
                else:
                    result_img[y, x] = result_img[y, left]
            else:  # Grayscale image
                if left < pattern_width:
                    result_img[y, x] = pattern[y % pattern.shape[0], left]
                else:
                    result_img[y, x] = result_img[y, left]
        
        # Fill in the pattern strip
        for x in range(pattern_width):
            if len(result_img.shape) == 3:  # Color
                result_img[y, x] = pattern[y % pattern.shape[0], x, :]
            else:  # Grayscale
                result_img[y, x] = pattern[y % pattern.shape[0], x]
    
    # Convert back to PIL Image
    if len(result_img.shape) == 3:
        return Image.fromarray(result_img, mode='RGB')
    else:
        return Image.fromarray(result_img, mode='L')

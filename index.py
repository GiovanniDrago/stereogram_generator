import os
import cv2
import sys
from matplotlib import pyplot as plt

# Add the lib folder to the Python path so we can import mask_gen, pattern_gen, and combine
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

# Import functions from the respective modules
from mask_gen import create_depth_mask
from pattern_gen import generate_pattern
from combine import create_stereogram

# Define input, working, and output folders
input_folder = 'input'
working_folder = 'working'
output_folder = 'output'

# Ensure working and output folders exist
os.makedirs(working_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    
    # Check if the file is an image (you can adjust extensions based on your needs)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Create the depth mask using the function from mask_gen.py
        depth_map = create_depth_mask(input_path)

        # Save the depth map to the working folder
        depth_output_path = os.path.join(working_folder, f'depth_{filename}')
        cv2.imwrite(depth_output_path, depth_map)

        # Generate a pattern using the generate_pattern function from pattern_gen.py
        pattern = generate_pattern(depth_map.shape[1], depth_map.shape[0])  # Use the same size as the depth map

        # Save the pattern to the working folder
        pattern_output_path = os.path.join(working_folder, f'pattern_{filename}')
        pattern.save(pattern_output_path)

        # Generate the stereogram using the depth map and pattern
        stereogram = create_stereogram(depth_map, pattern)

        # Save the stereogram to the output folder
        stereogram_output_path = os.path.join(output_folder, f'stereogram_{filename}')
        stereogram.save(stereogram_output_path)

        # Optionally: display the depth map, pattern, and stereogram (you can remove this part if not needed)
        plt.imshow(depth_map, cmap='gray')
        plt.title(f'Depth map for {filename}')
        plt.show()

        plt.imshow(pattern, cmap='gray')
        plt.title(f'Pattern for {filename}')
        plt.show()

        plt.imshow(stereogram, cmap='gray')
        plt.title(f'Stereogram for {filename}')
        plt.show()

print("All images processed and saved. Depth and pattern are in 'working', stereograms are in 'output'.")

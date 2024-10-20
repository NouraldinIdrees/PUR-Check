import os
import random
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def display_image_with_coco_annotations(image_path, annotations, color='red'):
    # Create a figure and axis to display the image
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Load the image using OpenCV and convert it from BGR (OpenCV default) to RGB color space for correct color display
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    ax.imshow(image)
    ax.axis('off')  # Turn off the axes for better visualization
    
    # Set the title of the plot to the image filename
    img_filename = os.path.basename(image_path)
    ax.set_title(img_filename, fontsize=12)

    # Get the image ID by matching the filename in the annotations
    img_id = next(item for item in annotations['images'] if item["file_name"] == img_filename)['id']
    
    # Filter the annotations to only include those related to the current image
    img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]

    # Iterate over each annotation to display the segmentation polygons
    for ann in img_annotations:
        for seg in ann['segmentation']:
            # Convert the segmentation points into a list of (x, y) tuples
            poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            # Create a polygon patch using Matplotlib
            polygon = patches.Polygon(poly, closed=True, edgecolor=color, fill=False)
            # Add the polygon to the axis
            ax.add_patch(polygon)

    # Adjust the layout to avoid any overlapping
    plt.tight_layout()
    # Display the final plot
    plt.show()

# Load COCO annotations from the specified JSON file
with open(r'D:\Studium\Master\Masterarbeit\Code\DL\COCO_output\train\coco_annotations.json', 'r') as f:
    annotations = json.load(f)

# Get all image file paths from the annotations
image_dir = r"D:\Studium\Master\Masterarbeit\Code\DL\COCO_output\train"
all_image_files = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]

# Select a random image file to display
random_image_file = random.choice(all_image_files)

# Display the selected image with the contour annotations
display_image_with_coco_annotations(random_image_file, annotations, color='red')
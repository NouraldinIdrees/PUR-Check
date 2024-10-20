import os
import cv2
import numpy as np
import json
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

def get_image_mask_pairs(image_dir, mask_dir):
    image_paths = []
    mask_paths = []
    
    # Iterate over all files in the image directory
    for file in os.listdir(image_dir):
        if file.endswith('.tif'):
            # Construct image and corresponding mask paths
            image_path = os.path.join(image_dir, file)
            mask_path = os.path.join(mask_dir, f"Mask of {file}")
            # If the corresponding mask exists, add the paths to the lists
            if os.path.exists(mask_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)
    
    return image_paths, mask_paths

def mask_to_polygons(mask, epsilon=1.0):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:  # Ensure the contour has enough points to form a valid polygon
            poly = contour.reshape(-1).tolist()
            if len(poly) > 4:  # Ensure the polygon has more than two points (valid shape)
                polygons.append(poly)
    return polygons

def process_data(image_paths, mask_paths, output_dir):
    annotations = []
    images = []
    image_id = 0
    ann_id = 0
    
    # Iterate over all image and mask pairs
    for img_path, mask_path in zip(image_paths, mask_paths):
        try:
            image_id += 1
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            # Read the mask
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")
            
            # Copy image to output directory
            shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
            
            # Add image information to the COCO dataset
            images.append({
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "height": img.shape[0],
                "width": img.shape[1]
            })
            
            # Get unique values in the mask (e.g., different object labels)
            unique_values = np.unique(mask)
            for value in unique_values:
                if value == 0:  # Ignore background value
                    continue
                
                # Create a binary mask for the current object label
                object_mask = (mask == value).astype(np.uint8) * 255
                # Extract polygons from the binary mask
                polygons = mask_to_polygons(object_mask)
                
                # Create an annotation for each polygon
                for poly in polygons:
                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,  # Assuming a single category (e.g., "hole")
                        "segmentation": [poly],  # Polygon coordinates
                        "area": cv2.contourArea(np.array(poly).reshape(-1, 2)),  # Area of the polygon
                        "bbox": list(cv2.boundingRect(np.array(poly).reshape(-1, 2))),  # Bounding box of the polygon
                        "iscrowd": 0  # Indicates whether the annotation is a crowd (0 = not a crowd)
                    })
        except Exception as e:
            # Print error and continue with the next pair if an error occurs
            print(e)
            continue
    
    # Create the COCO-style output dictionary
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "hole"}]  # Define category information
    }
    
    # Write the annotations to a JSON file in the output directory
    with open(os.path.join(output_dir, 'coco_annotations.json'), 'w') as f:
        json.dump(coco_output, f)

def process_images_in_parallel(image_paths, mask_paths, output_dir):
    # Use ThreadPoolExecutor to process images and masks in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(lambda p: process_data([p[0]], [p[1]], output_dir), zip(image_paths, mask_paths))

def main():
    # Define input and output directories
    image_dir = r'D:\Studium\Master\Masterarbeit\Code\DL\02_Cropped Images'
    mask_dir = r'D:\Studium\Master\Masterarbeit\Code\DL\03_Masks'
    output_dir = r'D:\Studium\Master\Masterarbeit\Code\DL\COCO_output'
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    # Create output directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get image and mask file paths
    image_paths, mask_paths = get_image_mask_pairs(image_dir, mask_dir)
    
    # Split data into training and validation sets
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    
    # Process training and validation data in parallel
    process_images_in_parallel(train_img_paths, train_mask_paths, train_dir)
    process_images_in_parallel(val_img_paths, val_mask_paths, val_dir)

if __name__ == '__main__':
    main()
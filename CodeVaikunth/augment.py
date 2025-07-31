import cv2
import os
import numpy as np
import random

# --- Configuration ---
# Define the paths to your images and labels
image_dir = os.path.join("data", "train", "images")
label_dir = os.path.join("data", "train", "labels")

# --- Helper Functions ---
def adjust_brightness(image, value=30):
    """Adjusts brightness by adding or subtracting a value."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Add or subtract the brightness value
    # Ensure the values stay within the 0-255 range
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def add_occlusion(image, size_percent=0.25):
    """Adds a random black rectangle to simulate occlusion."""
    h, w, _ = image.shape
    occ_h = int(h * size_percent)
    occ_w = int(w * size_percent)
    
    # Get a random top-left corner for the occlusion
    x1 = random.randint(0, w - occ_w)
    y1 = random.randint(0, h - occ_h)
    
    # Draw a black rectangle
    image[y1:y1+occ_h, x1:x1+occ_w] = (0, 0, 0)
    return image

# --- Main Script ---
print("Starting augmentation...")

# Loop through all the label files in the training directory
for label_filename in os.listdir(label_dir):
    if label_filename.endswith(".txt"):
        # Find the corresponding image file
        image_filename = label_filename.replace(".txt", ".png") # Change to .jpg if your images are JPGs
        image_path = os.path.join(image_dir, image_filename)
        label_path = os.path.join(label_dir, label_filename)

        if os.path.exists(image_path):
            # Read the image
            image = cv2.imread(image_path)
            
            # 1. Create a brightness augmented version
            bright_image = adjust_brightness(image, value=50) # Make images brighter
            bright_image_name = image_filename.replace(".png", "_aug_bright.png")
            cv2.imwrite(os.path.join(image_dir, bright_image_name), bright_image)
            # Copy the label file for the new image
            new_label_path = os.path.join(label_dir, bright_image_name.replace(".png", ".txt"))
            os.system(f"copy {label_path} {new_label_path}") # Use "cp" for Mac/Linux

            # 2. Create an occlusion augmented version
            occlusion_image = add_occlusion(image.copy()) # Use a copy of the original image
            occlusion_image_name = image_filename.replace(".png", "_aug_occlusion.png")
            cv2.imwrite(os.path.join(image_dir, occlusion_image_name), occlusion_image)
            # Copy the label file for the new image
            new_label_path = os.path.join(label_dir, occlusion_image_name.replace(".png", ".txt"))
            os.system(f"copy {label_path} {new_label_path}") # Use "cp" for Mac/Linux
            
print(f"Augmented {image_filename}")

print("Augmentation complete!")
print("You can now re-train your model.")
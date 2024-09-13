import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paths to data
train_images_dir = './data/training_images'
annotations_file = './data/train_solution_bounding_boxes.csv'

# Image dimensions (resize all images to this size)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Function to load and preprocess images
def load_images(image_dir, annotation_file):
    annotations = pd.read_csv(annotation_file)
    
    images = []
    boxes = []

    for idx, row in annotations.iterrows():
        image_path = os.path.join(image_dir, row['image'])
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {row['image']} not found.")
            continue
        
        # Resize image to a fixed size
        image_resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Normalize the image pixels to range [0, 1]
        image_resized = image_resized / 255.0

        # Append image and its bounding box to respective lists
        images.append(image_resized)
        
        # Normalize bounding box coordinates (dividing by image width and height)
        boxes.append([
            row['xmin'] / image.shape[1],  # Normalize by image width
            row['ymin'] / image.shape[0],  # Normalize by image height
            row['xmax'] / image.shape[1],  # Normalize by image width
            row['ymax'] / image.shape[0],  # Normalize by image height
        ])
    
    return np.array(images), np.array(boxes)

# Split data into training and validation sets
def split_data(images, boxes, test_size=0.2):
    return train_test_split(images, boxes, test_size=test_size, random_state=42)

if __name__ == "__main__":
    # Load and preprocess data
    images, boxes = load_images(train_images_dir, annotations_file)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = split_data(images, boxes)

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

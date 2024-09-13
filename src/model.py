import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.saving import register_keras_serializable

# Path to save the model
MODEL_SAVE_PATH = "models/object_detection_model.h5"

# Data directories and annotation file
TRAIN_DIR = "data/training_images/"
ANNOTATION_FILE = "data/train_solution_bounding_boxes.csv"

# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU is available: {tf.config.list_physical_devices('GPU')}")

# Custom loss function (Mean Squared Error)
@register_keras_serializable()
def mse(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()
    return mse_loss(y_true, y_pred)

def load_annotations(annotation_file):
    """
    Load bounding box annotations from the CSV file.
    """
    print("Loading annotations...")
    annotations = pd.read_csv(annotation_file)
    return annotations

def load_images_and_boxes(train_dir, annotations):
    """
    Load the images and corresponding bounding boxes.
    """
    print("Loading images and bounding boxes...")
    images = []
    bounding_boxes = []
    
    for idx, row in annotations.iterrows():
        img_path = os.path.join(train_dir, row['image'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]  # Get the image height and width
            img = cv2.resize(img, (224, 224))  # Resize image to 224x224
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            
            # Bounding boxes (xmin, ymin, xmax, ymax) need to be resized proportionally
            bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            scale_w, scale_h = 224 / img_w, 224 / img_h
            bbox_scaled = [bbox[0] * scale_w, bbox[1] * scale_h, bbox[2] * scale_w, bbox[3] * scale_h]
            bounding_boxes.append(bbox_scaled)
    
    return np.array(images), np.array(bounding_boxes)

def split_data(images, bounding_boxes):
    """
    Split the data into training and validation sets.
    """
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(images, bounding_boxes, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def build_model(input_shape):
    """
    Build the object detection model.
    """
    print("Building the model...")
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='linear')  # Output layer for bounding boxes (xmin, ymin, xmax, ymax)
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=mse)
    print("Model built successfully!")
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the model.
    """
    print("Starting model training...")

    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

    print("Model training completed!")

    # Save the trained model
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved at: {MODEL_SAVE_PATH}")

    return history

if __name__ == "__main__":
    print("Starting the object detection project...")

    # Load annotations
    annotations = load_annotations(ANNOTATION_FILE)

    # Load images and bounding boxes
    images, bounding_boxes = load_images_and_boxes(TRAIN_DIR, annotations)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(images, bounding_boxes)

    # Build the model
    input_shape = (224, 224, 3)
    model = build_model(input_shape)

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)

    print("Object detection project completed successfully!")

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.saving import register_keras_serializable

# Path to load the model
MODEL_PATH = "models/object_detection_model.h5"

# Path to test images
TEST_DIR = "data/testing_images/"

# Annotation file for test data (if available)
TEST_ANNOTATION_FILE = "data/test_solution_bounding_boxes.csv"

# Custom loss function (Mean Squared Error)
@register_keras_serializable()
def mse(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()
    return mse_loss(y_true, y_pred)

def load_test_images(test_dir):
    """
    Load the test images.
    """
    print("Loading test images...")
    test_images = []
    image_names = []
    
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # Resize to match training input
            img = img / 255.0  # Normalize pixel values
            test_images.append(img)
            image_names.append(img_name)
    
    return np.array(test_images), image_names

def evaluate_model(model, test_images):
    """
    Evaluate the model on test images.
    """
    print("Evaluating model on test images...")
    predictions = model.predict(test_images)
    return predictions

def visualize_predictions(test_images, predictions, image_names):
    """
    Visualize predictions by drawing bounding boxes on images.
    """
    for i in range(len(test_images)):
        img = (test_images[i] * 255).astype(np.uint8)  # Convert back to original pixel scale
        xmin, ymin, xmax, ymax = predictions[i]
        
        # Draw bounding box on image
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        
        # Display the image with bounding box
        cv2.imshow(f"Prediction - {image_names[i]}", img)
        cv2.waitKey(0)  # Wait for a key press to proceed

    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Loading the trained model...")
    
    # Load the model with custom objects
    model = load_model(MODEL_PATH, custom_objects={'mse': mse})

    # Load test images
    test_images, image_names = load_test_images(TEST_DIR)

    # Evaluate the model
    predictions = evaluate_model(model, test_images)

    # Visualize predictions
    visualize_predictions(test_images, predictions, image_names)

    print("Model evaluation completed successfully!")

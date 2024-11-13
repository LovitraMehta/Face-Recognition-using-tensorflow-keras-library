import os
import numpy as np
import cv2
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from PIL import Image

# Disable GPU to avoid CUDA warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define paths
MODEL_PATH = "C:/Users/91981/OneDrive/Documents/Facerecognition ML/facenet_keras.h5"
LFW_DATASET_PATH = r"C:\Users\91981\OneDrive\Documents\Facerecognition ML\lfw"
PROCESSED_DATA_PATH = "C:/Users/91981/OneDrive/Documents/Facerecognition ML/processed_faces"

# Load MTCNN for face detection
detector = MTCNN()

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load the FaceNet model
print("Loading FaceNet model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# Function to extract a face from an image
def extract_face(filename, required_size=(160, 160)):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    if not results:
        return None
    x, y, width, height = results[0]['box']
    x, y = max(0, x), max(0, y)
    face = image[y:y + height, x:x + width]
    face_image = Image.fromarray(face).resize(required_size)
    return np.asarray(face_image)

# Function to prepare the dataset
def prepare_dataset():
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    for person_name in os.listdir(LFW_DATASET_PATH):
        person_folder = os.path.join(LFW_DATASET_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue

        person_processed_folder = os.path.join(PROCESSED_DATA_PATH, person_name)
        if not os.path.exists(person_processed_folder):
            os.makedirs(person_processed_folder)

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            face = extract_face(image_path)
            if face is None:
                continue

            processed_image_path = os.path.join(person_processed_folder, image_name)
            Image.fromarray(face).save(processed_image_path)
            print(f"Processed: {processed_image_path}")

# Prepare the dataset
prepare_dataset()
print("Dataset preparation completed.")

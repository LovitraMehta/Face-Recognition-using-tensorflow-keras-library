import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import joblib

# Disable GPU to avoid CUDA warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define paths
MODEL_PATH = r"C:\Users\91981\OneDrive\Documents\Facerecognition ML\facenet_keras.h5\facenet_keras.h5"
PROCESSED_DATA_PATH = "C:/Users/91981/OneDrive/Documents/Facerecognition ML/processed_faces"
CLASSIFIER_PATH = "C:/Users/91981/OneDrive/Documents/Facerecognition ML/face_classifier.pkl"

# Load the FaceNet model
model = load_model(MODEL_PATH, compile=False)

# Function to get embeddings
def get_embedding(face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    embedding = model.predict(face)[0]
    return embedding

# Prepare data for training
X, y = [], []
for person_name in os.listdir(PROCESSED_DATA_PATH):
    person_folder = os.path.join(PROCESSED_DATA_PATH, person_name)
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        face = img_to_array(image)
        embedding = get_embedding(face)
        X.append(embedding)
        y.append(person_name)

X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train SVM classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, y_encoded)

# Save the classifier
joblib.dump((classifier, label_encoder), CLASSIFIER_PATH)
print("Model training completed and saved.")

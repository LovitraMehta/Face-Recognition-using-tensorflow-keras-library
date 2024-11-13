import os
import cv2
import numpy as np
import joblib
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Disable GPU to avoid CUDA warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define paths
MODEL_PATH = "C:/Users/91981/OneDrive/Documents/Facerecognition ML/facenet_keras.h5"
CLASSIFIER_PATH = "C:/Users/91981/OneDrive/Documents/Facerecognition ML/face_classifier.pkl"

# Load the FaceNet model and the classifier
model = load_model(MODEL_PATH, compile=False)
classifier, label_encoder = joblib.load(CLASSIFIER_PATH)

# Load MTCNN detector
detector = MTCNN()

# Function to recognize face
def recognize_face(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    for result in results:
        x, y, w, h = result['box']
        face = frame[y:y + h, x:x + w]
        face_image = Image.fromarray(face).resize((160, 160))
        face_array = img_to_array(face_image)
        embedding = model.predict(np.expand_dims(face_array, axis=0))[0]

        prediction = classifier.predict([embedding])
        probability = classifier.predict_proba([embedding])[0]
        label = label_encoder.inverse_transform(prediction)[0]
        confidence = max(probability)

        if confidence > 0.7:
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

# Start video stream
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = recognize_face(frame)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

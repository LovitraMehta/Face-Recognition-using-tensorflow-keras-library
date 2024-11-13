Face Recognition ML Project
This project is a machine learning-based face recognition system designed to detect faces, classify them as friendly or unfriendly, and alert the user via email if an unfamiliar face is detected. It leverages TensorFlow for the deep learning model, MTCNN for face detection, and OpenCV for capturing video from the camera.

Table of Contents
Features
Project Structure
Requirements
Installation
Usage
Dataset
How It Works
References
Features
Detect faces using MTCNN.

Classify detected faces as friendly or unfriendly using a pre-trained FaceNet model (facenet_keras.h5).
Update the database of known faces by capturing new friendly faces via the GUI.
Alert via email with a photo of the detected unfamiliar face.

Project Structure
text
Copy code
Facerecognition ML/
├── venv/                  # Virtual environment folder
├── lfw/                   # LFW dataset for training
├── captured_faces/        # Folder to store captured face images
├── facenet_keras.h5       # Pre-trained FaceNet model
├── prepdb.py              # Script to prepare the face database
├── training.py            # Script to train the classifier
├── facerecognition.py     # Main application script
├── requirements.txt       # List of required Python libraries
├── README.md              # Project documentation (this file)
└── email_alert.py         # Script for sending email alerts

Requirements
Python 3.8 or higher
A working webcam or external camera
Internet connection for email alerts (SMTP setup)

Libraries
The following Python libraries are required:
tensorflow
mtcnn
numpy
opencv-python-headless
scikit-learn
smtplib

Installation
Step 1: Clone the Repository
bash git clone https://github.com/your-username/Facerecognition-ML.git
cd Facerecognition-ML

Step 2: Set Up a Virtual Environment
bash python -m venv venv
.\venv\Scripts\Activate

Step 3: Install Dependencies
bash pip install -r requirements.txt4

Step 4: Download the Pre-trained Model
Place the facenet_keras.h5 file in the project directory. You can download it from Google Drive link.

Step 5: Dataset Setup
Ensure the LFW (Labeled Faces in the Wild) dataset is extracted to the lfw/ folder. The structure should look like:

text
lfw/
├── person1/
│   ├── image1.jpg
│   └── image2.jpg
├── person2/
│   ├── image1.jpg
│   └── image2.jpg
└── ...
Usage

Step 1: Prepare the Database
Run the script to prepare the embeddings of known faces:
bash python prepdb.py

Step 2: Train the Classifier

Train the model using the extracted embeddings:
bash python training.py

Step 3: Start Face Recognition
Run the main application to start recognizing faces:

bash python facerecognition.py
Email Alerts
If an unfamiliar face is detected, an email will be sent to the specified address (mlovitra@gmail.com) with a photo of the detected face.

How It Works
Face Detection: The MTCNN library detects faces in real-time from the camera feed.
Feature Extraction: A pre-trained FaceNet model (facenet_keras.h5) extracts facial features as embeddings.
Classification: The classifier compares the embeddings against the database of known faces.
Email Alert: If an unfamiliar face is detected, an alert email with a photo is sent.
Troubleshooting
Unicode Error: If you encounter a UnicodeError, use raw strings by adding r before the file paths.
TensorFlow Errors: Ensure you have a compatible TensorFlow version installed (tensorflow==2.9.1 is recommended).
Email Not Sent: Check your SMTP settings and allow less secure apps if necessary.

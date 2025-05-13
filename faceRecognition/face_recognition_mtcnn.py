# bash
# pip install mtcnn opencv-python numpy tensorflow keras
# pip install deepface


import cv2
import os
from deepface import DeepFace
from mtcnn import MTCNN

# Load known faces and names
def load_known_faces(folder):
    known_faces = []
    known_names = []

    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            name = os.path.splitext(filename)[0]
            known_faces.append(img)
            known_names.append(name)
    return known_faces, known_names

# Recognize face using DeepFace
def recognize_face(face_img, known_faces, known_names):
    try:
        result = DeepFace.find(img_path=face_img, db_path='known_faces', enforce_detection=False, model_name='Facenet')
        if len(result[0]) > 0:
            return os.path.splitext(os.path.basename(result[0].iloc[0]['identity']))[0]
        else:
            return "Unknown"
    except Exception as e:
        print("Recognition error:", e)
        return "Unknown"

# Detect and recognize faces in the test image
def process_image(image_path):
    img = cv2.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for face in faces:
        x, y, w, h = face['box']
        cropped_face = img[y:y+h, x:x+w]
        face_file = 'temp_face.jpg'
        cv2.imwrite(face_file, cropped_face)

        name = recognize_face(face_file, known_faces, known_names)

        # Draw box and name
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, name, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Labeled Faces", img)
    filename = 'result_mtnn.jpg'
    cv2.imwrite(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run
known_faces, known_names = load_known_faces('known_faces')
process_image('group_pics/grp11.jpg')


# Documentation
# Create the known_faces/ folder and add one image per person.

# File name = person's name (e.g., alice.jpg).

# Add test.jpg with unknown faces.

# Run the script above.

# You’ll see the output image with face boxes and names!
# Use clear, well-lit images for best accuracy.

# You can switch Facenet to VGG-Face, ArcFace, etc. in the DeepFace.find() call.

# You can add confidence thresholds too.

# Print the FaceNet Model Layers
# You can load the actual FaceNet model and print the summary like this:
# from deepface.basemodels import Facenet

# # Load the FaceNet model
# model = Facenet.loadModel()

# # Print the layers
# model.summary()

# This will show:

# Each layer name

# Layer type (Conv2D, Dense, etc.)

# Output shape

# Number of parameters



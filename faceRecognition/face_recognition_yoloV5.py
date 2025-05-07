# bash
# pip install torch torchvision opencv-python deepface
# git clone https://github.com/deepcam-cn/yolov5-face.git
# cd yolov5-face
# pip install -r requirements.txt
# face_yolo_project/
# ├── known_faces/
# │   ├── alice.jpg
# │   ├── bob.jpg
# ├── yolov5-face/
# │   ├── detect.py
# │   └── weights/yolov5s-face.pt  ← pretrained YOLOv5-face model
# ├── test.jpg  ← group photo
import torch
import cv2
import os
from deepface import DeepFace
import numpy as np
import sys
sys.path.append('yolov5-face') 
from models.experimental import attempt_load

# Load YOLOv5-face model
#model = torch.hub.load('deepcam-cn/yolov5-face', 'custom', path='yolov5-face/weights/yolov5s-face.pt', source='github')
#model = torch.hub.load('yolov5-face', 'yolov5s-face.pt', source='local')
# Path to your weights
weights_path = 'yolov5-face/yolov5s-face.pt'

# Load the model
#model = DetectMultiBackend(weights_path, device='cpu')
model = attempt_load(weights_path, map_location='cpu')
model.conf = 0.5  # confidence threshold
# Print model summary
print(model)
# Print layers, parameters, and shapes of the model
for name, layer in model.named_modules():
    print(f"Layer name: {name}, Layer type: {layer.__class__.__name__}")
    for param_name, param in layer.named_parameters():
        print(f"  Parameter name: {param_name}, Shape: {param.shape}")

# Recognize face using DeepFace
def recognize_face(face_img_path, db_path='known_faces'):
    try:
        result = DeepFace.find(img_path=face_img_path, db_path=db_path, enforce_detection=False, model_name='Facenet')
        if len(result[0]) > 0:
            return os.path.splitext(os.path.basename(result[0].iloc[0]['identity']))[0]
        else:
            return "Unknown"
    except Exception as e:
        print("Recognition error:", e)
        return "Unknown"

# Detect and recognize
def detect_and_label(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    faces = results.xyxy[0]  # x1, y1, x2, y2, conf, landmarks...

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face[:4])
        face_crop = img[y1:y2, x1:x2]

        temp_path = f'temp_face_{i}.jpg'
        cv2.imwrite(temp_path, face_crop)

        name = recognize_face(temp_path)

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, name, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        os.remove(temp_path)

    cv2.imshow("YOLOv5-Face Recognition", img)
    filename = 'result_Yolov5.jpg'
    cv2.imwrite(filename, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run
detect_and_label("group_pics/grp11.jpg")

# Output
# YOLOv5-face will detect faces in test.jpg.

# Each face will be matched against the known_faces database using FaceNet (via DeepFace).

# The image will display with names labeled on each face.

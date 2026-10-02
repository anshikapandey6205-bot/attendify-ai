import cv2
import os
import numpy as np

DATASET_DIR = "uploads/faces"

os.makedirs(DATASET_DIR, exist_ok=True)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    return faces


def save_face_samples(student_id, frame):
    faces = detect_faces(frame)

    if len(faces) == 0:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]

    filename = os.path.join(
        DATASET_DIR,
        f"{student_id}.jpg"
    )

    cv2.imwrite(filename, face)

    return True


def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    images = []
    labels = []

    for filename in os.listdir(DATASET_DIR):

        if not filename.endswith(".jpg"):
            continue

        path = os.path.join(DATASET_DIR, filename)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        student_id = int(
            filename.replace(".jpg", "")
        )

        images.append(image)
        labels.append(student_id)

    if not images:
        return None

    recognizer.train(
        images,
        np.array(labels)
    )

    return recognizer


def recognize_face(frame, recognizer):

    if recognizer is None:
        return None, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(frame)

    for x, y, w, h in faces:

        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        # LBPH confidence is distance-like:
        # lower generally means a closer match.
        if confidence < 80:

            return label, confidence, (x, y, w, h)

    return None, None, None
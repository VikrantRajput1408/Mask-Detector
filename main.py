# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blob = faceNet.detectMultiScale(gray, 1.1, 4)
    faces = []
    locs = []
    preds = []

    for (x, y, w, h) in blob:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)/255.0
        faces.append(face)
        locs.append((x, y, x+w, y+h))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

faceNet = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

maskNet = load_model("mask.h5")
LABEL = os.listdir("data/")

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)

LABEL = os.listdir("data/")
print(LABEL)
while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        print(round(pred[0], 2))

        label = LABEL[np.argmax(pred)]

        if label == "with_mask":
            color = (0, 255, 0)
        elif label == "without_mask":
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        label = "{}:{:.2f}%".format(label, max(pred) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

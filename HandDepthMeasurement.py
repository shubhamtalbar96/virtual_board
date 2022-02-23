import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(maxHands=1, detectionCon=0.85)

# Find Function
# x is the raw distance and y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coefficients = np.polyfit(x, y, 2)

# Loop
while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)

    if hands:
        landmark_list = hands[0]['lmList']
        x, y, w, h = hands[0]['bbox']
        print(landmark_list)

        x1, y1, z1 = landmark_list[5]
        x2, y2, z2 = landmark_list[17]

        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        A, B, C = coefficients
        distance_in_cm = A*distance**2 + B*distance + C

        # print(distance, distance_in_cm)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 3)
        cvzone.putTextRect(img, f'{int(distance_in_cm)} cm', (x, y))

    cv2.imshow("Image", img)
    cv2.waitKey(1)

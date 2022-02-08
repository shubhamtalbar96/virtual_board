import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
prev_time = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img, True)
    landmark_list = detector.find_position(img, hand_number=0, draw=True)

    # if len(landmark_list) != 0:
    #     print(landmark_list[0])

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

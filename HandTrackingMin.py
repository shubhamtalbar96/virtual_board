import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

prev_time = 0
current_time = 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for index, land_mark in enumerate(hand_landmarks.landmark):
                # print(index, land_mark)
                height, width, channel = img.shape
                center_x, center_y = int(land_mark.x * width), int(land_mark.y * height)
                # print(index, center_x, center_y)

                if index == 4:
                    cv2.circle(img, (center_x, center_y), 25, (255, 0, 255), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

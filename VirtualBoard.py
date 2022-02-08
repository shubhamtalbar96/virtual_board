import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folder_path = "Header"
my_list = os.listdir(folder_path)
# print(my_list[1])

overlay_list = []
file_path = 'Header/header_1.png'
overlay_list.append(cv2.imread(file_path))

file_path = 'Header/header_2.png'
overlay_list.append(cv2.imread(file_path))

file_path = 'Header/header_3.png'
overlay_list.append(cv2.imread(file_path))

file_path = 'Header/header_4.png'
overlay_list.append(cv2.imread(file_path))

file_path = 'Header/header_5.png'
overlay_list.append(cv2.imread(file_path))

# print(len(overlay_list))
header = overlay_list[0]
# print(header)

draw_color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(min_detection_confidence=0.85)

brush_thickness = 10
eraser_thickness = 100

x_previous = 0
y_previous = 0

image_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:
        # print(landmark_list)

        # tip of the index finger
        x1, y1 = landmark_list[8][1:]

        # tip of the middle finger
        x2, y2 = landmark_list[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingers_up()
        # print(fingers)

        # 4. If Selection Mode - Two fingers are up
        if fingers[1] and fingers[2]:
            print("Selection Mode")

            x_previous = 0
            y_previous = 0

            # if selecting something in the header
            if y1 < 125:
                if 0 < x1 < 300:
                    header = overlay_list[1]
                    print("Selected 1")
                    draw_color = (255, 0, 255)
                elif 320 < x1 < 620:
                    header = overlay_list[2]
                    print("Selected 2")
                    draw_color = (255, 0, 0)
                elif 640 < x1 < 940:
                    header = overlay_list[3]
                    print("Selected 3")
                    draw_color = (0, 255, 0)
                elif 960 < x1 < 1280:
                    header = overlay_list[4]
                    print("Selected 4")
                    draw_color = (0, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), draw_color, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            print("Drawing Mode")

            if x_previous == 0 and y_previous == 0:
                x_previous, y_previous = x1, y1

            if draw_color == (0, 0, 0):
                cv2.line(img, (x_previous, y_previous), (x1, y1), draw_color, eraser_thickness)
                cv2.line(image_canvas, (x_previous, y_previous), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (x_previous, y_previous), (x1, y1), draw_color, brush_thickness)
                cv2.line(image_canvas, (x_previous, y_previous), (x1, y1), draw_color, brush_thickness)
            x_previous, y_previous = x1, y1

    image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
    _, image_inverse = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)
    image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, image_inverse)
    img = cv2.bitwise_or(img, image_canvas)

    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, image_canvas, 0.5, 0)

    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", image_canvas)
    cv2.waitKey(1)

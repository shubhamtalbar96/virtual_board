import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
import TesseractOcr as tocr


overlay_list = []
file_path = 'Panel/mainPanel.jpg'
overlay_list.append(cv2.imread(file_path))

file_path = 'Panel/drawPanel.jpg'
overlay_list.append(cv2.imread(file_path))

file_path = 'Panel/erasePanel.jpg'
overlay_list.append(cv2.imread(file_path))

file_path = 'Panel/recoPanel.jpg'
overlay_list.append(cv2.imread(file_path))

# print(len(overlay_list))
header = overlay_list[0]
# print(header)

font = cv2.FONT_HERSHEY_SIMPLEX
draw_color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(min_detection_confidence=0.85)
text_recognizer = tocr

brush_thickness = 10
eraser_thickness = 100

x_previous = 0
y_previous = 0

image_canvas = np.zeros((720, 1280, 3), np.uint8)
flag = False

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
            # Save the canvas if it is not empty
            if flag:
                images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
                filename = os.path.join(images_folder, "input_sample.jpg")
                cv2.imwrite(filename, image_inverse)

            print("Selection Mode")

            x_previous = 0
            y_previous = 0

            # if selecting something in the header
            if y1 < 125:
                if 0 < x1 < 400:
                    header = overlay_list[1]
                    print("Selected Drawing Mode")
                    draw_color = (255, 0, 255)
                elif 450 < x1 < 850:
                    header = overlay_list[2]
                    print("Selected Erasing Mode")
                    draw_color = (0, 0, 0)
                elif 900 < x1 < 1280:
                    header = overlay_list[3]
                    print("Selected Recognition Mode")
                    if flag:
                        print("Recognizing hand drawn image")
                        text_result = text_recognizer.detect_character()
                        print(text_result)
                    else:
                        print("Nothing to recognize !!")

            cv2.rectangle(img, (x1, y1), (x2, y2), draw_color, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            flag = True

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
    # cv2.imshow("Gray Canvas", image_gray)
    cv2.waitKey(1)

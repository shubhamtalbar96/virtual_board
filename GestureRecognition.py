import asyncio
import base64
import cv2
import numpy as np
import time
import os
import HandTrackingModule as hand_tracking_module
import TesseractModule as tess_ocr

from websocket import create_connection
class GestureOcr:
    def __init__(self):
        self.overlay_list = []
        file_path = 'Panel/mainPanel.jpg'
        self.overlay_list.append(cv2.imread(file_path))

        file_path = 'Panel/drawPanel.jpg'
        self.overlay_list.append(cv2.imread(file_path))

        file_path = 'Panel/erasePanel.jpg'
        self.overlay_list.append(cv2.imread(file_path))

        file_path = 'Panel/recoPanel.jpg'
        self.overlay_list.append(cv2.imread(file_path))

        # print(len(overlay_list))
        self.header = self.overlay_list[0]

        # stroke color for mid-air sketching
        self.draw_color = (255, 0, 255)

        self.brush_thickness = 8
        self.eraser_thickness = 100
        self.drawing_thickness = 15

    def recognize_gesture(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 1280)
        cap.set(4, 720)

        detector = hand_tracking_module.HandDetector(min_detection_confidence=0.85)
        text_recognizer = tess_ocr.Ocr()

        x_previous = 0
        y_previous = 0

        image_canvas = np.zeros((720, 1280, 3), np.uint8)
        flag = False
        try:
            ws = 1
            ws = create_connection("ws://localhost:9999/")
        except Exception as e:
            print(e, 'create_connection')

        while True:
            # 1. Import image
            success, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.resize(image, (1280, 720)) 
            # 2. Find Hand Landmarks
            image = detector.find_hands(image)
            landmark_list = detector.find_position(image, draw=False)
            

            if len(landmark_list) != 0:
                # print(landmark_list)

                # tip of the index finger
                x1, y1 = landmark_list[8][1:]

                # tip of the middle finger
                x2, y2 = landmark_list[12][1:]

                # 3. Check which fingers are up
                fingers = detector.fingers_up()
                # print(fingers)

                # 4. If Clear Mode - Clean the entire canvas
                if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 \
                        and fingers[3] == 1 and fingers[4] == 1:
                    # print("Cleaning Mode")

                    # reset flag value to False
                    flag = False
                    # reset the entire image canvas
                    image_canvas = np.zeros((720, 1280, 3), np.uint8)
                    x_previous = 0
                    y_previous = 0

                # 4. If Selection Mode - Two fingers are up
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 \
                        and fingers[3] == 0 and fingers[4] == 0:
                    # print("Selection Mode")

                    # Save the canvas if it is not empty
                    if flag:
                        images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
                        filename = os.path.join(images_folder, "input_sample.jpg")
                        cv2.imwrite(filename, image_inverse)

                    x_previous = 0
                    y_previous = 0

                    # if selecting something in the header
                    if y1 < 125:
                        if 0 < x1 < 450:
                            self.header = self.overlay_list[1]
                            print("Selected Drawing Mode")
                            self.draw_color = (255, 0, 255)

                        elif 450 < x1 < 850:
                            self.header = self.overlay_list[2]
                            print("Selected Erasing Mode")
                            self.draw_color = (0, 0, 0)

                        elif 900 < x1 < 1280:
                            self.header = self.overlay_list[3]
                            print("Selected Recognition Mode")

                            if flag:
                                print("Recognizing hand drawn image")
                                recognition_result = text_recognizer.detect_character(image)
                                # print(recognition_result)
                                ws.close()
                                return recognition_result
                            else:
                                print("Nothing to recognize !!")

                    cv2.rectangle(image, (x1, y1), (x2, y2), self.draw_color, cv2.FILLED)

                # 5. If Drawing Mode - Index finger is up
                elif fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and \
                        fingers[3] == 0 and fingers[4] == 0:
                    # print("Drawing Mode")
                    self.header = self.overlay_list[1]

                    # set flag value to true if drawing something
                    flag = True

                    cv2.circle(image, (x1, y1), self.drawing_thickness, self.draw_color, cv2.FILLED)

                    if x_previous == 0 and y_previous == 0:
                        x_previous, y_previous = x1, y1

                    if self.draw_color == (0, 0, 0):
                        cv2.line(image, (x_previous, y_previous), (x1, y1), self.draw_color, self.eraser_thickness)
                        cv2.line(image_canvas, (x_previous, y_previous), (x1, y1), self.draw_color,
                                 self.eraser_thickness)
                    else:
                        cv2.line(image, (x_previous, y_previous), (x1, y1), self.draw_color, self.brush_thickness)
                        cv2.line(image_canvas, (x_previous, y_previous), (x1, y1), self.draw_color,
                                 self.brush_thickness)

                    x_previous, y_previous = x1, y1

                # 6. If Recognition Mode - First and last fingers are up
                elif fingers[1] == 1 and fingers[4] == 1 and fingers[0] == 0 \
                        and fingers[2] == 0 and fingers[3] == 0:
                    # print("Selected Recognition Mode")
                    self.header = self.overlay_list[3]

                    if flag:
                        print("Recognizing hand drawn image")
                        recognition_result = text_recognizer.detect_character(image)
                        # print(recognition_result)
                        ws.close()
                        return recognition_result
                    else:
                        print("Nothing to recognize !!")

                else:
                    # print("Default")
                    pass

            image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
            _, image_inverse = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)
            image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)
            image = cv2.bitwise_and(image, image_inverse)
            image = cv2.bitwise_or(image, image_canvas)

            image[0:125, 0:1280] = self.header
            # image = cv2.addWeighted(image, 0.5, image_canvas, 0.5, 0)

            #cv2.imshow("Image", image)
            sendImage(ws, image)
            cv2.waitKey(1)
        
def sendImage(ws, image):
    try:
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)        
        ws.send(jpg_as_text)        
    except Exception as e:
        print(e, 'sendImage')
        ws.close()
        return 0

def main():
    gesture_ocr = GestureOcr()

    while True:
        reply = gesture_ocr.recognize_gesture()
        print(f"Detected self: {reply}")


if __name__ == "__main__":
    main()

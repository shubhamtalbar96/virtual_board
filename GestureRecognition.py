import base64
import cv2
import numpy as np
import os
import HandTrackingModule as hand_tracking_module
import TesseractModule as tess_ocr
import Constants

from websocket import create_connection


def send_image(web_socket, image):
    """Send encoded image via web socket to Unity backend server"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        web_socket.send(jpg_as_text)
    except Exception as exception:
        print(exception, 'exception in send_image method')
        # web_socket = create_connection("ws://localhost:9999/")


class GestureOcr:
    def __init__(self):
        # import custom made images to create a selection panel
        # to be embedded in the virtual board
        self.overlay_list = []
        file_path = 'Panel/mainPanel.jpg'
        self.overlay_list.append(cv2.imread(file_path))

        file_path = 'Panel/drawPanel.jpg'
        self.overlay_list.append(cv2.imread(file_path))

        file_path = 'Panel/erasePanel.jpg'
        self.overlay_list.append(cv2.imread(file_path))

        file_path = 'Panel/recoPanel.jpg'
        self.overlay_list.append(cv2.imread(file_path))

        # set header initially to main/empty panel
        self.header = self.overlay_list[0]

        # stroke color for mid-air sketching
        self.draw_color = (255, 0, 255)

        # define thickness for different brushes
        self.brush_thickness = 8
        self.eraser_thickness = 100
        self.drawing_thickness = 15

    def recognize_gesture(self):
        """Recognize gestures/numerical digits drawn mid-air on the virtual board"""

        # setting video capture for windows vs mac machine
        if Constants.IS_WINDOWS_MACHINE:
            cap = cv2.VideoCapture(int(Constants.VIDEO_SOURCE), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(int(Constants.VIDEO_SOURCE))
        cap.set(3, 1280)
        cap.set(4, 720)

        detector = hand_tracking_module.HandDetector(min_detection_confidence=0.85)
        text_recognizer = tess_ocr.Ocr()

        # set the previous x and y coordinates for drawing strokes initially to origin
        x_previous = 0
        y_previous = 0

        # create an empty canvas
        # this is a different plane to be added to real-time video feed
        # Gestures are drawn on to this canvas and used for image detection & recognition
        image_canvas = np.zeros((720, 1280, 3), np.uint8)

        # flag used to identify if anything is drawn on image canvas
        flag = False

        # open a web socket connection to server running in Unity project
        # this is to send the video feed from Open-CV backend to Unity backend
        # as a continuous byte stream
        try:
            if Constants.IS_PIP_MODE:
                web_socket = create_connection("ws://localhost:"+str(Constants.UNITY_SERVER_PORT_NUMBER)+"/")
        except Exception as exception:
            print(exception, 'Exception occurred while creating connection in GestureRecognition module')
            # return 1 by default
            return '1'

        # run a continuous loop to capture video feed from camera source
        while True:
            # 1. Import image
            success, image = cap.read()

            # flip image to assist a user while drawing
            image = cv2.flip(image, 1)
            image = cv2.resize(image, (1280, 720)) if Constants.IS_WINDOWS_MACHINE else image

            # 2. Find Hand Landmarks
            image = detector.find_hands(image)

            # find the landmark positions on the detected hand in video feed
            landmark_list = detector.find_position(image, draw=False)

            # proceed only if any landmark detected
            # i.e. if user is waving hands in front of camera
            if len(landmark_list) != 0:
                # print(landmark_list)

                # coordinates for tip of the index finger
                x1, y1 = landmark_list[8][1:]

                # coordinates for tip of the middle finger
                x2, y2 = landmark_list[12][1:]

                # 3. Check which fingers are up
                fingers = detector.fingers_up()
                # print(fingers)

                # 4.1 If Clear Mode
                if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 \
                        and fingers[3] == 1 and fingers[4] == 1:
                    # Clean the entire image canvas
                    # if all fingers are up
                    # print("Clear Mode")

                    # reset flag value to False
                    # since there is nothing on the canvas to detect
                    flag = False

                    # reset the entire image canvas
                    image_canvas = np.zeros((720, 1280, 3), np.uint8)

                    # reset previous x and y coordinate values
                    x_previous = 0
                    y_previous = 0

                # 4.2 If Selection Mode
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 \
                        and fingers[3] == 0 and fingers[4] == 0:
                    # Index and Middle fingers are up
                    # print("Selection Mode")

                    # Save the canvas if it is not empty
                    # if flag:
                    #     images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
                    #     filename = os.path.join(images_folder, "input_sample.jpg")
                    #     cv2.imwrite(filename, image_inverse)

                    # reset previous x and y coordinate values
                    x_previous = 0
                    y_previous = 0

                    # if selecting something in the header
                    if y1 < int(Constants.PANEL_HEIGHT):
                        # overlay panel header with the appropriate
                        # panel image based on the x coordinate
                        # 4.2.1 Selected Drawing Mode
                        if 0 < x1 < 450:
                            self.header = self.overlay_list[1]
                            self.draw_color = (255, 0, 255)
                            print("Selected Drawing Mode")

                        # 4.2.2 Selected Erasing Mode
                        elif 450 < x1 < 900:
                            self.header = self.overlay_list[2]
                            self.draw_color = (0, 0, 0)
                            print("Selected Erasing Mode")

                        # 4.2.3 Selected Recognition Mode
                        elif 900 < x1 < 1280:
                            self.header = self.overlay_list[3]
                            print("Selected Recognition Mode")

                            # check flag to assure if the image canvas is non-empty
                            if flag:
                                print("Recognizing hand drawn image")
                                recognition_result = text_recognizer.detect_character(image_inverse)
                                # print(recognition_result)

                                if Constants.IS_PIP_MODE:
                                    web_socket.close()
                                return recognition_result
                            else:
                                print("Empty Canvas, nothing to recognize !!")

                    # display a rectangle between index and middle finger tips
                    cv2.rectangle(image, (x1, y1), (x2, y2), self.draw_color, cv2.FILLED)

                # 5. If Drawing Mode
                elif fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and \
                        fingers[3] == 0 and fingers[4] == 0:
                    # If Index finger is up
                    # print("Drawing Mode")
                    self.header = self.overlay_list[1]

                    # set flag value to true since user is drawing something
                    flag = True

                    # display a marker at the tip of index finger
                    cv2.circle(image, (x1, y1), self.drawing_thickness, self.draw_color, cv2.FILLED)

                    # set x_previous and y_previous to current coordinates of index finger
                    if x_previous == 0 and y_previous == 0:
                        x_previous, y_previous = x1, y1

                    # If Eraser is selected
                    if self.draw_color == (0, 0, 0):
                        cv2.line(image, (x_previous, y_previous), (x1, y1), self.draw_color, self.eraser_thickness)
                        cv2.line(image_canvas, (x_previous, y_previous), (x1, y1), self.draw_color,
                                 self.eraser_thickness)
                    else:
                        cv2.line(image, (x_previous, y_previous), (x1, y1), self.draw_color, self.brush_thickness)
                        cv2.line(image_canvas, (x_previous, y_previous), (x1, y1), self.draw_color,
                                 self.brush_thickness)

                    # update x_previous & y_previous coordinates
                    x_previous, y_previous = x1, y1

                # 6. If Recognition Mode - First and last fingers are up
                elif fingers[1] == 1 and fingers[4] == 1 and fingers[0] == 0 \
                        and fingers[2] == 0 and fingers[3] == 0:
                    # print("Selected Recognition Mode")
                    self.header = self.overlay_list[3]

                    # check flag to assure if the image canvas is non-empty
                    if flag:
                        print("Recognizing hand drawn image")
                        recognition_result = text_recognizer.detect_character(image_inverse)
                        # print(recognition_result)

                        if Constants.IS_PIP_MODE:
                            web_socket.close()
                        return recognition_result
                    else:
                        print("Empty Canvas, nothing to recognize !!")

            # convert BGR image to GRAY image
            image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)

            # threshold the gray image
            _, image_inverse = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)

            # convert the binary image to BGR
            image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)

            # take bitwise and of original image & binary image
            image = cv2.bitwise_and(image, image_inverse)
            image = cv2.bitwise_or(image, image_canvas)

            # pin the panel header to the top of camera
            image[0:125, 0:1280] = self.header

            # cv2.imshow("Image", image)
            # send image frame-by-frame to Unity backend server
            # this is to enable picture in picture mode within Unity
            if Constants.IS_PIP_MODE:
                send_image(web_socket, image)
            cv2.waitKey(1)


def main():
    gesture_ocr = GestureOcr()

    while True:
        reply = gesture_ocr.recognize_gesture()
        print(f"Detected self: {reply}")


if __name__ == "__main__":
    main()

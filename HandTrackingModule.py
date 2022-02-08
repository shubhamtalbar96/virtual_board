import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.75,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_image_mode, self.max_num_hands,
                                         self.model_complexity, self.min_detection_confidence,
                                         self.min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.landmark_list = None
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[0]
            for index, land_mark in enumerate(my_hand.landmark):
                # print(index, land_mark)
                height, width, channel = img.shape
                center_x, center_y = int(land_mark.x * width), int(land_mark.y * height)
                # print(index, center_x, center_y)
                self.landmark_list.append([index, center_x, center_y])
                if draw:
                    cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

        return self.landmark_list

    def fingers_up(self):
        fingers = []

        # Thumb
        if self.landmark_list[self.tip_ids[0]][1] < self.landmark_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for index in range(1, 5):
            if self.landmark_list[self.tip_ids[index]][2] < self.landmark_list[self.tip_ids[index] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    prev_time = 0

    while True:
        success, img = cap.read()
        img = detector.find_hands(img, True)
        landmark_list = detector.find_position(img, hand_number=0, draw=True)

        if len(landmark_list) != 0:
            print(landmark_list[0])

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

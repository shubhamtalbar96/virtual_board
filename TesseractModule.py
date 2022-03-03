import os
import cv2
import pytesseract
import Constants


class Ocr:
    def __init__(self):
        self.images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        self.filename = os.path.join(self.images_folder, "input_sample.jpg")


    # gray scale
    def gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # blur
    def blur(self, img):
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        return img_blur

    # threshold
    def threshold(self, img):
        # pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
        img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        # cv2.imshow('threshold', img)
        return img

    # text detection
    def contours_text(self, orig, image_canvas, contours):
        output_text = ""
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Drawing a rectangle on copied image
            # rect = cv2.rectangle(image_canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # cv2.imshow('cnt', rect)
            # cv2.waitKey()

            # Cropping the text block for giving input to OCR
            cropped = orig[y:y + h, x:x + w]
            cropped = cv2.copyMakeBorder(
                cropped, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            cv2.imwrite(self.filename, cropped)
            # cv2.imshow('cropped', cropped)

            text = pytesseract.image_to_string(
                cropped, config=("-c tessedit"
                                 "_char_whitelist=0123456789"
                                 " --psm 10"
                                 " -l osd"
                                 " "))

            print(f"Contour Text detected as: {text}")
            output_text = output_text + text
        return output_text

    def detect_character(self, image_canvas):
        # Finding contours
        im_gray = self.gray(image_canvas)
        im_blur = self.blur(im_gray)
        im_thresh = self.threshold(im_blur)

        if Constants.IS_WINDOWS_MACHINE:
            pytesseract.pytesseract.tesseract_cmd =r'C:/Program Files/Tesseract-OCR/tesseract.exe'

        contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return self.contours_text(im_thresh, image_canvas, contours)

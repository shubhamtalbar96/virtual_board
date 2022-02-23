import os
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image


class Ocr:
    def __init__(self):
        self.images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        self.filename = os.path.join(self.images_folder, "input_sample.jpg")

    def detect_character(self):
        with PyTessBaseAPI(psm=PSM.SINGLE_CHAR) as api:
            image = Image.open(self.filename)
            api.SetImage(image)
            detected_character = api.GetUTF8Text()
            return detected_character

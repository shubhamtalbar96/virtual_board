import os
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image


def detect_character():
    with PyTessBaseAPI(psm=PSM.SINGLE_CHAR) as api:
        images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        filename = os.path.join(images_folder, "input_sample.jpg")

        image = Image.open(filename)
        api.SetImage(image)
        detected_character = api.GetUTF8Text()
        # print(detected_character)
        return detected_character

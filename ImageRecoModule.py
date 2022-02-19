from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
from PIL import Image

import os
import sys
import time


class ImageRecognizer:
    def __init__(self,
                 subscription_key="d24630ddf376411f9a105aba25b26d7b",
                 endpoint="https://cs291a-westus.cognitiveservices.azure.com/",
                 ):
        self.subscription_key = subscription_key
        self.endpoint = endpoint
        self.images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        self.cv_client = ComputerVisionClient(self.endpoint, CognitiveServicesCredentials(self.subscription_key))

    def recognize_image(self):
        print("===== Running AZURE Image Reco API =====")
        # Get image path
        read_image_path = os.path.join(self.images_folder, "input_sample.jpg")
        # Open the image
        read_image = open(read_image_path, "rb")

        # Call API with image and raw response (allows you to get the operation location)
        read_response = self.cv_client.read_in_stream(read_image, raw=True)
        # Get the operation location (URL with ID as last appendage)
        read_operation_location = read_response.headers["Operation-Location"]
        # Take the ID off and use to get results
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for the retrieval of the results
        while True:
            read_result = self.cv_client.get_read_result(operation_id)
            if read_result.status.lower() not in ['notstarted', 'running']:
                break
            print('Waiting for result...')
            time.sleep(5)

        # Print results, line by line
        output_text = ""
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    output_text = output_text + line.text
                    print(line.text)
                    print(line.bounding_box)

        print("End of Image recognition")
        return output_text

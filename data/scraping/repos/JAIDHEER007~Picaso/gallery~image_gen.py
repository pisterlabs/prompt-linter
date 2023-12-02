import openai

import json
import uuid
import os
import base64

from .awsS3 import AwsManager

from django.conf import settings
from feature.models import Switch

import logging
logger = logging.getLogger('watchtower-logger')

openai.api_key = settings.OPENAI_API_KEY
class ImageGen:
    openaiImage = openai.Image
   
    def __init__(self, prompt):
        if prompt is None or len(prompt) == 0:
            raise ValueError("Prompt cannot be empty")
        
        self.__prompt = prompt

    def __callOpenAI(self) -> dict:
        dummyImageSwitch, reposne = True, None
        dummyImageSwitch = Switch.objects.get(feature_name = 'dummy_image').toggle
        logger.info(f"Dummy Switch {dummyImageSwitch}")
        if not dummyImageSwitch:
            logger.info("Inside Open AI Image API")
            response = ImageGen.openaiImage.create(
                prompt = self.__prompt,
                n = 1,
                size = "512x512",
                response_format = "b64_json"
            )
        else:
            logger.info("Inside Dummy Image Response")
            with open(os.path.join(settings.BASE_DIR, 'media', 'dummy_response.json'), 'r') as fileHandle:
                response = json.load(fileHandle)

        if response is None:
            logger.error("Response object is None")
            raise Exception("Reponse object is None")
        
        return response['data'][0]
    
    def __base64toPNG(self, base64str) -> None:
        self.__imgName = uuid.uuid4().hex[:10]
        self.__imgFile =  self.__imgName + '.png'

        logger.info("Started decoding B64 Img")
        with open(os.path.join(settings.BASE_DIR, 'media', self.__imgFile), "wb") as fileHandle:
            fileHandle.write(base64.decodebytes(bytes(base64str, "utf-8")))

    def moveToS3(self) -> None:
        res = AwsManager.upload_file(
            filename = os.path.join(settings.BASE_DIR, 'media', self.__imgFile), 
            key = self.__imgFile
        )

        logger.info(f"File Upload Status {res}")

        # Removing the file
        os.remove(os.path.join(settings.BASE_DIR, 'media', self.__imgFile))
        logger.info("File Removed")

    def execute(self) -> str:
        logger.info("Image Gen Execute Function Started")
        responseJson = self.__callOpenAI()
        self.__base64toPNG(responseJson['b64_json'])
        self.moveToS3()

        return self.__imgName
    
            

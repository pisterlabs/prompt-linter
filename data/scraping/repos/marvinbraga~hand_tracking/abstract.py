import base64
import json
from abc import ABCMeta, abstractmethod
from datetime import datetime
from enum import Enum

import openai
from decouple import config


class TextEngineEnum(Enum):
    DAVINCI_003 = "text-davinci-003"


class AbstractOpenAi(metaclass=ABCMeta):
    openai.api_key = config("open_ai_key", cast=str)

    def __init__(self, prompt=None):
        self._prompt = prompt
        self._result = None

    @property
    def result(self):
        return self._result

    @abstractmethod
    def execute(self):
        pass


class AbstractImageOpenAI(AbstractOpenAi):
    def __init__(self, prompt=None, path="./data/", size="256x256", number_of_images=1):
        super().__init__(prompt)
        self._size = size
        self._number_of_images = number_of_images
        self._path = path
        self._file_name = None

    def _get_name(self):
        text = ""
        if self._prompt:
            text = "_" + self._prompt.replace(" ", "_")
        filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S%f")}{text}'.strip()[0:70]
        return filename

    def _save_image(self, response):
        image_64_encode = json.loads(str(response))["data"][0]["b64_json"]
        image_64_decode = base64.b64decode(image_64_encode)
        self._file_name = self._path + self._get_name() + ".png"
        with open(self._file_name, "wb") as image_result:
            image_result.write(image_64_decode)
        return self

    @property
    def file_name(self):
        return self._file_name

    def execute(self):
        pass

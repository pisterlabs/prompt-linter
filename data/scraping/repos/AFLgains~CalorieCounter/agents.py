from abc import ABC, abstractmethod
from prompts import CALORIE_ESTIMATOR_SYSTEM_PROMPT, RESPONSE_ENCODER_SYSTEM_PROMPT
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

class Agent(ABC):

    @property
    @abstractmethod
    def system_prompt(self):
        """Abstract property for system_prompt."""
        pass

    @abstractmethod
    def run(self):
        pass

class CalorieEstimator(Agent):
    def __init__(self, key):
        self.client = OpenAI(api_key = key)
           
    def system_prompt(self):
        return CALORIE_ESTIMATOR_SYSTEM_PROMPT
    
    def encode_image(self, image_file):
        img = Image.open(image_file)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def run(self, picture,details = None):
        if details is None:
            details = "None"
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt()
                }
                ,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", 
                        "text": f"Additional Details: {details}. Estimate the calories and macronutrient (protien, fats and carbs) breakdown. "},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{self.encode_image(picture)}",
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response



class ResponseEncoder(Agent):
    def __init__(self, key):
        self.client = OpenAI(api_key = key)
           
    def system_prompt(self):
        return RESPONSE_ENCODER_SYSTEM_PROMPT
    

    def run(self,response_text):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format = {"type": "json_object"},
            messages=[
                {"role": "system","content": self.system_prompt()},
                {"role": "user", "content": response_text}
            ]
        )
        return response
    
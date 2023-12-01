import base64
import requests
from io import BytesIO
from PIL import Image
from langchain.prompts import PromptTemplate
import os

# OpenAI API Key
api_key = os.environ['OPENAI_API_KEY']


ROBOT_PROMPT = """You are a robot dog, and this is what your eyes see right now. 
It is not an image or a picture. """

EXPLORE_PROMPT = """Describe what you can see, in as much detail as possible and with the direction you would need to walk to get there. 
Start your answer with with 'I see ' """

DIRECTION_PROMPT = """ This is what you can see right now. There is a {item}. Please return the direction you would need to walk to get there, and the Distance. Answer with a JSON object like this: {{"direction": "left", "distance": "10 cm"}}"""

def explore_image(image: Image) -> str:
  """You can see what is around you."""
  return analyze_image(image, EXPLORE_PROMPT)

def get_direction_from_image(image: Image, item: str) -> str:
  """Get the direction to reach an item"""
  prompt_template = PromptTemplate.from_template(DIRECTION_PROMPT)
  prompt = prompt_template.render(item=item)
  return analyze_image(image, prompt)

def analyze_image(image: Image, prompt: str)-> str:
  """Analyses an image and returns a description"""
  full_prompt = ROBOT_PROMPT + prompt
  buffered = BytesIO()
  image.save(buffered, format="JPEG")
  base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": full_prompt,
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  result = response.json()['choices'][0]['message']['content']

  return(result)

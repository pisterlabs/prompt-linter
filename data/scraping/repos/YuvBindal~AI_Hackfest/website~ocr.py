import easyocr
import PIL
from PIL import ImageDraw
import numpy as np
import openai
from dotenv import load_dotenv, find_dotenv
import os


def read_image(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    im = PIL.Image.open(image_path)
    im_np = np.array(im)
    bounds = reader.readtext(im_np)
    return bounds
"""
reader = easyocr.Reader(['en'], gpu=False)
im = PIL.Image.open("/RECIEPT1.png")
im_np = np.array(im)
bounds = reader.readtext(im_np)

def draw_boxes(image, bounds, color='red', width=2):
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    
    return image

#draw_boxes(im, bounds)

"""

def bounds_to_text(bounds):
    receipt_text = ""
    for i in range(len(bounds)):
        receipt_text+= bounds[i][1]
        receipt_text += "\n"
    return receipt_text

"""
receipt_text = ""
for i in range(len(bounds)):
    receipt_text+= bounds[i][1]
    receipt_text += "\n"
#print(receipt_text)
"""



def gpt_convert(receipt_text):
    # Set up the OpenAI API
    load_dotenv(find_dotenv())
    api_key = os.environ.get("API_KEY")

    openai.api_key = api_key

    # Call the API
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "user", 
            "content": f"Given the following text from a receipt, kindly identify what items are bought and for how much. Display it in the format [Product Name] : [Price]. Do not include anything else.\n{receipt_text}"}
            ]
    )

    gpt_response = response['choices'][0]['message']['content']

    return gpt_response

"""
# Set up the OpenAI API
load_dotenv(find_dotenv())
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

# Call the API
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "user", 
         "content": f"Given the following text from a receipt, kindly identify what items are bought and for how much. Display it in the format [Product Name] : [Price]. Do not include anything else.\n{receipt_text}"}
        ]
)

gpt_response = response['choices'][0]['message']['content']

"""


#split the gpt_response by line split, further split by : and then store in a dictionary
def text_to_dict(text):
    text = text.split("\n")
    text = [i.split(":") for i in text]
    final_dict = {i[0].strip():i[1].strip() for i in text}
    return final_dict


def ocr_main(image_path) :
    bounds = read_image(image_path)
    receipt_text = bounds_to_text(bounds)
    gpt_response = gpt_convert(receipt_text)
    dict_response = text_to_dict(gpt_response)
    return dict_response

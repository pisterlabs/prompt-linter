import openai
import os
from PIL import Image
import pytesseract

openai.api_key = "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"

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = ''
image = Image.open(image_path)

word = pytesseract.image_to_string(image)
print(word)


response = openai.Image.create(
  prompt="drawing of a " + word,
  n=1,
  size="200x200"
)
image_url = response['data'][0]['url']
print(image_url)

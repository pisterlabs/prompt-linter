# python code
!pip install pytesseract
!apt install tesseract-ocr -y
!apt install libtesseract-dev

# python code
import cv2
import pytesseract

# Define the path
image_path = 'D:\colab_pro\AUTOGEN\groupchat\IvV2y.png'

# Read the image
img = cv2.imread(image_path)

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Tesseract to extract text
text = pytesseract.image_to_string(gray)

# Print the extracted text
print(text)


#--------------------- Above code is just to run pytesseract in colab ---------------------#

!pip install pytesseract
!apt install tesseract-ocr -y
!apt install libtesseract-dev


import cv2
import pytesseract

! pip install langchain unstructured[all-docs] pydantic lxml openai chromadb tiktoken
!apt-get install -y poppler-utils


#------------
from typing import Any
import os
from unstructured.partition.pdf import partition_pdf
import os

input_path = os.getcwd()
output_path = os.path.join(os.getcwd(), "output")

# Get elements
raw_pdf_elements = partition_pdf(
    filename=os.path.join(input_path, "test.pdf"),
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=output_path,
)

#------------

import base64

text_elements = []
table_elements = []
image_elements = []

# Function to encode images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

for element in raw_pdf_elements:
    if 'CompositeElement' in str(type(element)):
        text_elements.append(element)
    elif 'Table' in str(type(element)):
        table_elements.append(element)

table_elements = [i.text for i in table_elements]
text_elements = [i.text for i in text_elements]

# Tables
print(len(table_elements))

# Text
print(len(text_elements))


#--------

for image_file in os.listdir(output_path):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(output_path, image_file)
        encoded_image = encode_image(image_path)
        image_elements.append(encoded_image)
print(len(image_elements))

#--------   

!pip install python-dotenv openai
import os, openai
from dotenv import load_dotenv

# Get API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#----------

from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

chain_gpt_35 = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
chain_gpt_4_vision = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

#-------

# Function for text summaries
def summarize_text(text_element):
    prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

# Function for table summaries
def summarize_table(table_element):
    prompt = f"Summarize the following table:\n\n{table_element}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

# Function for image summaries
def summarize_image(encoded_image):
    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the contents of this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = chain_gpt_4_vision.invoke(prompt)
    return response.content

#--------

# Processing table elements with feedback and sleep
table_summaries = []
for i, te in enumerate(table_elements[0:2]):
    summary = summarize_table(te)
    table_summaries.append(summary)
    print(f"{i + 1}th element of tables processed.")
    
#--------

# Processing text elements with feedback and sleep
text_summaries = []
for i, te in enumerate(text_elements[0:2]):
    summary = summarize_text(te)
    text_summaries.append(summary)
    print(f"{i + 1}th element of texts processed.")

#--------

# Processing image elements with feedback and sleep
image_summaries = []
for i, ie in enumerate(image_elements[0:2]):
    summary = summarize_image(ie)
    image_summaries.append(summary)
    print(f"{i + 1}th element of images processed.")







import openai
import pytesseract
from PIL import Image
import difflib
import os
from dotenv import load_dotenv

# Function to extract text from an image using OCR
def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))
# Function to compare two strings and return a similarity score

def text_similarity(string1, string2):
    return difflib.SequenceMatcher(None, string1, string2).ratio()

# Function to process multiple images and extract text
def process_multiple_images(image_paths):
    extracted_texts = {}
    for idx, image_path in enumerate(image_paths):
        # Extract text from the image
        image_text = extract_text_from_image(image_path)
        extracted_texts[f"Image {idx + 1}"] = image_text
    return extracted_texts

# Function to call ChatGPT API and get the model's response for a given prompt
MODEL = 'gpt-3.5-turbo'

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pass


def call_chatgpt():


    response = openai.ChatCompletion.create(

        model="gpt-3.5-turbo",

        messages= [ {"role": "system",

                "content": f"""G

                For Every Image , refine the text extracted from every image and if there is any error or spelling mistake , correct it by your knowledge  but correct only 'words' and

'not' values of it .

Also give "quantity" with it's value as it is given in extracted text, "rate" with it's value as it is given in extracted text and ("amount = quantity*rate") in the last as note. Give nothing else other than this \n{extracted_text}\n"""

    }],

        max_tokens=150,

        temperature=0.7

        

    )

    return response['choices'][0]['message']['content']

 

# List of paths to the images with the petrol bills

image_paths = ["1.png"

    # Add more image paths as needed

]

 

# Process multiple images and extract text

extracted_text_dict = process_multiple_images(image_paths)

 

# Print the extracted text for each image

for image_name, extracted_text in extracted_text_dict.items():

    print(f"{image_name}:")

    print(extracted_text)

    print("----------------------")

 

    # Call ChatGPT API with the extracted text as a prompt

    

    

    chatgpt_response = call_chatgpt()

 

    # Print ChatGPT response

    print("ChatGPT response:")

    print(chatgpt_response)

    print("----------------------")
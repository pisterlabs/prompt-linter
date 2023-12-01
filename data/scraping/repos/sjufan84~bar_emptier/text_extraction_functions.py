# %%
import streamlit as st
import os
import pdfplumber
import requests
from dotenv import load_dotenv
load_dotenv()
import openai
from google.oauth2 import service_account
from spellchecker import SpellChecker


# Set your API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])


# Function to extract the text from a PDF file
def extract_pdf(pdf_file):
    # Here we are going to use the pdfplumber library to extract the text from the PDF file
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# %%
# Define a function to extract the text from an image using the Google Vision API
from google.cloud import vision

def detect_document(uploaded_image):

    client = vision.ImageAnnotatorClient(credentials=credentials)

    #with io.BytesIO(uploaded_image) as image_file:
    #    content = image_file

    #image = vision.Image(uploaded_image)

    response = client.document_text_detection(image=uploaded_image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(
                    paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))
    
    response_text = response.full_text_annotation.text


    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    return response_text

# %%

# Define a function to run the extracted text through a spellchecker
def spellcheck_text(text):

    # Load the custom domain-specific list
    with open("./resources/new_ingredients.txt", "r") as file:
        cooking_terms = [line.strip() for line in file]

    # Initialize the spell-checker
    spell = SpellChecker(language='en')
    spell.word_frequency.load_words(cooking_terms)

    # Tokenize the returned text from the Vision model`)
    tokens = text.split()

    # Correct the misspelled words
    corrected_tokens = []
    for token in tokens:
        if token not in cooking_terms:
            corrected = spell.correction(token)
            if corrected:
                corrected_tokens.append(corrected)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)

    # Reconstruct the corrected text
    corrected_text = ' '.join(corrected_tokens)

    return corrected_text



def extract_text_from_txt(file):
    # Extract text from a text file
    return file.read()

# We need two functions for feeding extracted text to the OpenAI API -- 1 for text and pdf that uses GPT 3.5 turbo, and one for photots that uses GPT 4.
# The extracted text from photos generally needs to be cleaned up a bit more and needs a more powerful model to handle it.

def text_menu_edit(menu):
    # Use the OpenAI API to re-format the menu

    # Use the OpenAI API to re-format the menu

    messages = [
        {
            "role": "system",
            "content": "You are an amazingly helpful assistant restauranteur who edits user's menus in a format to make it readable,\
                if necessary, and be able to answer questions about it.."
        },
        {
            "role": "user",
             "content" : f"""
             Can you help me format this menu {menu} to make it more readable, if needed, and to be able to answer questions about it?  Please stay as true to the original menu {menu} as possible.
            """
        },
    ]


           # Use the OpenAI API to generate the menu
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = messages,
            max_tokens=750,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            temperature=0.6,
            n=1,
            top_p =1
        )
        edited_menu = response.choices[0].message.content

    except (requests.exceptions.RequestException, openai.error.APIError):

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = messages,
                max_tokens=750,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                temperature=0.6,
                n=1,
                top_p =1
            )
            edited_menu = response.choices[0].message.content
        except (requests.exceptions.RequestException, openai.error.APIError):

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages = messages,
                max_tokens=750,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                temperature=0.6,
                n=1,
                top_p =1
            )
            edited_menu = response.choices[0].message.content
       
    return edited_menu



def photo_menu_edit(menu):
    # Use the OpenAI API to re-format the menu

    messages = [
        {
            "role": "system",
            "content": "You are an amazingly helpful assistant restauranteur who edits user's menus to make them more readable."
        },
        {
            "role": "user",
             "content" : f"""
             Can you help me format this menu {menu} to make it as readable as possible?  Please stay as true to the original menu {menu} as possible.
            """
        },
    ]

           # Use the OpenAI API to generate a menu
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = messages,
            max_tokens=750,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            temperature=0.6,
            n=1,
            top_p =1
        )
        edited_menu = response.choices[0].message.content

    except (requests.exceptions.RequestException, openai.error.APIError):

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-0314",
                messages = messages,
                max_tokens=750,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                temperature=0.6,
                n=1,
                top_p =1
            )
            edited_menu = response.choices[0].message.content
        except (requests.exceptions.RequestException, openai.error.APIError):

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = messages,
                max_tokens=750,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                temperature=0.6,
                n=1,
                top_p =1
            )
            edited_menu = response.choices[0].message.content
       
    return edited_menu



def extract_and_concatenate_text(menu_files, menu_text_area):
    allowed_image_types = ["image/jpeg", "image/png", "image/jpg"]
    full_menu_text = ""

    for menu_file in menu_files:
        if menu_file.type == "application/pdf":
            menu_text = extract_pdf(menu_file)
        elif menu_file.type == "text/plain":
            menu_text = extract_text_from_txt(menu_file)
        elif menu_text_area != "":
            menu_text = menu_text_area
        elif menu_file.type in allowed_image_types:
            menu_text = detect_document(menu_file)
            menu_text = spellcheck_text(menu_text)
        else:
            st.write(f"Unsupported file type: {menu_file.type}")
            continue

        full_menu_text += menu_text + "\n\n"

    return full_menu_text


def edit_menu(full_menu_text, menu_files):
    allowed_image_types = ["image/jpeg", "image/png", "image/jpg"]
    last_uploaded_file = menu_files[-1]

    if last_uploaded_file.type in allowed_image_types:
        return photo_menu_edit(full_menu_text)
    else:
        return full_menu_text



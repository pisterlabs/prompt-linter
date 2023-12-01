import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')


# Function to extract text from an image using pytesseract
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)


# Function to convert PDF to images and extract text using pytesseract
def extract_text_from_pdf(pdf_path):
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)

    extracted_text = ''

    for image in images:
        # Preprocess image for better OCR results
        image = image.convert('RGB')
        image = image.resize((image.width * 3, image.height * 3), resample=Image.BOX)

        # Extract text from the image
        text = extract_text_from_image(image)
        extracted_text += text

    return extracted_text


# Function to extract PAN number using GPT-3
def extract_details(text):
    prompt = '''Your task is to parse through and find the following details from the text extracted from a travel bill present in triple backticks and return the them in json format. 
    If detail is not found, return None. (if there are multiple tickets in the provided text so return a single JSON object containing multiple JSON objects where each object represents a different travel expense).
    Incase of round trip tickets, count the travel expense just once and make the amount of the return trip to be 0. return JSON objects in the follwing format:

    {
        from: {
            place,
            date: (Day, DD/MM/YYYY), 
            time
        }, 
        to: {
            place,
            date: (Day, DD/MM/YYYY),
            time
        },
        mode of travel: (flight, cab, train, other),
        pnr no,
        amount: (in INR)
    }
    '''

    prompt += '```' + text + '```'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract PAN number from GPT-3 response
    completion_text = response.choices[0].message.content.strip()

    return completion_text


# Main function
def main():
    # Path to the PDF file
    pdf_path = './air.pdf'

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)

    print(extracted_text)
    print("--------------------")

    # Extract PAN number using GPT-3
    details_json = extract_details(extracted_text)

    # Print the extracted PAN number
    print(details_json)


if __name__ == '__main__':
    main()

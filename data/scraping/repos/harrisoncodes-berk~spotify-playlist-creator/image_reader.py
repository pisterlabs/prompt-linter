import pytesseract
from PIL import Image
import openai
from dotenv import dotenv_values

# Path to the image file
image_path = "./images/lineup.jpeg"

# Open the image file using PIL (Python Imaging Library)
image = Image.open(image_path)

# Perform OCR using pytesseract
extracted_text = pytesseract.image_to_string(image)

openai.api_key = dotenv_values(".env")["OPENAI_API_KEY"]

def ask_chat_gpt(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    )

    # Extract the assistant's reply from the API response
    assistant_reply = response["choices"][0]["message"]["content"]
    return assistant_reply

question = (
    "Please extract the artist names from the following text in the format (name1 - name2 - name3) Please respond without any additional text: \n\n"
    + extracted_text
)
response = ask_chat_gpt(question)
print(response)

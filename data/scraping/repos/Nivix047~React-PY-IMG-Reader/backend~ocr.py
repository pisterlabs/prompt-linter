import pytesseract
from PIL import Image
import requests
from io import BytesIO
import sys
import openai
import os

# Update this path based on your system
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


def summarize_text(text):
    """
    This function receives a text string, uses the OpenAI GPT-3.5 Turbo model to summarize it,
    and returns the generated summary. Labeled as: Text summarization function

    :param text: str, The text to summarize
    :return: str, The generated summary
    """
    # OpenAI API Key
    openai.api_key = os.getenv("OPENAI_KEY")

    # Generate a summary of the text using OpenAI GPT-3.5 Turbo
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Summarize in human readable language the text of the following document: {text}"}]
    )

    # Return the generated summary
    return completion.choices[0].message.content


# Function to extract text from image
def extract_text_from_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    text = pytesseract.image_to_string(img)
    return text


# Main execution
if __name__ == "__main__":
    try:
        url = sys.stdin.read().strip()
        text = extract_text_from_image(url)  # Extract text from image
        summary = summarize_text(text)  # Summarize extracted text
        print(summary)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

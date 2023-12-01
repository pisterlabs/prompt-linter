import io
from google.cloud import vision
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

client = vision.ImageAnnotatorClient()


def ocr(byte_stream: io.BytesIO) -> list:

    image = vision.Image(content=byte_stream.read())

    text = client.text_detection(image=image)

    raw_text = [t.description for t in text.text_annotations]

    return raw_text

def parse_ocr(raw_text: list, context: str) -> str:
    """
    Parses the raw text from the OCR function.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages = [
            {"role": "user", "content": f"OCR results: {raw_text}\n\n---\n\nVery briefly use the ocr results to response to the query in a way that is suitable for a voice response. Omit text that may be irrelevant and just there due to ocr inaccuracies: {context}"}
        ]
    )

    return response.choices[0].message.content

def parse_double_ocr(full_byte_stream: io.BytesIO, target_byte_stream: io.BytesIO, context):
    """
    Parses the raw text from the OCR function.
    """

    raw_text1 = ocr(full_byte_stream)
    raw_text2 = ocr(target_byte_stream)

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages = [
            {"role": "system", "content": "You are receiving two ocr results, the full and the target. The full is the entire image while the target is a small subset of the image meant to show the general area of interest. Never only rely on the target area"},
            {"role": "user", "content": f"OCR results: {raw_text1}\n\nTarget OCR results: {raw_text2}\n\n---\n\nVery briefly use the ocr results to response to the query in a way that is suitable for a voice response. Omit text that may be irrelevant and just there due to ocr inaccuracies: {context}"}
        ]
    )

    return response.choices[0].message.content

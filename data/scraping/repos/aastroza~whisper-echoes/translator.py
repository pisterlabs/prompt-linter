from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import instructor
from pydantic import BaseModel, Field, create_model

client = instructor.patch(OpenAI())

class TextDetail(BaseModel):
    language: str = Field(...,
                            description="Language of the input in ISO 639-1 code format")

def get_text_details(text):
    message = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.0,
        response_model=TextDetail,
        messages=[
            {"role": "user", "content": text},
        ]
    )

    return message

def get_text_translation(text, message, languages=['en', 'es']):

    for lang in languages:
        if lang != message.language:
            target_language = lang

    TextTranslation = create_model(
        'TextTranslation',
        translation=(str, Field(..., description="Translation of the input in the target language")),
        target_language=(str, f'{target_language}'),
        __base__=TextDetail,
    )
    translation = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.0,
        response_model=TextTranslation,
        messages=[
            {"role": "user", "content": text},
        ]
    )

    return translation

def translate(text, languages=['en', 'es']):
    message = get_text_details(text)
    translation = get_text_translation(text, message, languages)

    return translation.translation
# python imports
import os

# installed imports
import openai
from dotenv import load_dotenv


load_dotenv()

LOG_DIR = os.getenv("CHATBOT_LOG")
TEMP_FOLDER = os.getenv("TEMP_FOLDER")
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER")
WHATSAPP_TEMPLATE_NAMESPACE = os.getenv("WHATSAPP_TEMPLATE_NAMESPACE")
WHATSAPP_TEMPLATE_NAME = os.getenv("WHATSAPP_TEMPLATE_NAME")
CONTEXT_LIMIT = int(os.getenv("CONTEXT_LIMIT"))
WHATSAPP_CHAR_LIMIT = int(os.getenv("WHATSAPP_CHAR_LIMIT"))


class TimeoutError(Exception):
    # custom timeout class to handle ChatGPT timeout
    pass


## Vonage functions ---------------------------------------------------
def text_response(prompt: str, number: str) -> tuple:
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=2000,
        temperature=0.7,
        user=f"{str(number)}",
    )

    text = completion.choices[0].text
    tokens = int(completion.usage.total_tokens)

    return text, tokens


def send_text(
    client,
    text: str,
    to: str,
):
    return client.messages.send_message(
        {
            "channel": "whatsapp",
            "message_type": "text",
            "to": to,
            "from": WHATSAPP_NUMBER,
            "text": text,
        }
    )


def send_audio(client, audio_url, to):
    return client.messages.send_message(
        {
            "channel": "whatsapp",
            "message_type": "audio",
            "to": to,
            "from": WHATSAPP_NUMBER,
            "audio": {
                "url": audio_url,
            },
        }
    )


def send_image(client, image_url, to):
    return client.messages.send_message(
        {
            "channel": "whatsapp",
            "message_type": "image",
            "to": to,
            "from": WHATSAPP_NUMBER,
            "image": {"url": image_url},
        }
    )


def send_otp(
    client,
    to_number: str,
    otp: str,
):
    return client.messages.send_message(
        {
            "channel": "whatsapp",
            "message_type": "template",
            "to": to_number,
            "from": WHATSAPP_NUMBER,
            "template": {
                "name": f"{WHATSAPP_TEMPLATE_NAMESPACE}:{WHATSAPP_TEMPLATE_NAME}",
                "parameters": [
                    otp,
                ],
            },
            "whatsapp": {"policy": "deterministic", "locale": "en-GB"},
        }
    )

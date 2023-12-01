from openai import OpenAI
import os
import base64
from datetime import date

OPENAI_KEY = os.environ['OPENAI_KEY']

def get_audio(audio_64, format):
    client = OpenAI(
        api_key=OPENAI_KEY
    )

    # input_file = open("audio.txt", "r")
    audio_file_decoded = base64.b64decode(audio_64)
    # input_file.close()

    output_file = open("audio."+format, "wb")
    output_file.write(audio_file_decoded)
    output_file.close()

    audio_file = open("audio."+format, "rb")

    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        response_format="text",
        # language="it"
    )

    audio_file.close()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "I need an array of dict for each product, its product catgeory and the shelf_life and the day. The dictionary should only have as keys: product, category in english, shelf_life is just the number of days, date = 'n/a'. I need to generate the array of dict for these products:"+transcript+". I only need the array, please don't generate other text. Don't say anything else."}
        ]
    )
    return completion.choices[0].message.content

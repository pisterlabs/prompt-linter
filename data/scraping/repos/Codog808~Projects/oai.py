# import openai
from dotenv import load_dotenv
from httpx import Response
load_dotenv()
from openai import OpenAI
client = OpenAI()

# with open("openai.ky") as f:
#     client.api_key = f.read().strip()
def gpt():
    prompto = input("prompt: ")
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "you are a master at creating condensed code within python, an amazingly descriptive teacher of python also."},
                {"role": "user", "content": prompto}
                ]
            )

    # Extract the generated text from the response
    generated_text = response.choices[0].message.content

    print("Generated text:", generated_text)

def tts():
    from pathlib import Path
    file_path = Path(__file__).parent / "jpoken.mp3"
    response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=input("give me some text to turn into speech: ")
            )
    response.stream_to_file(file_path)

tts()

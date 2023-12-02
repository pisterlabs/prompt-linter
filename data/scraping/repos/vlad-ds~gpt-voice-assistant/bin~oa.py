import os

from dotenv import load_dotenv
import openai

load_dotenv()
GPT_MODEL = "gpt-3.5-turbo"
WHISPER_MODEL = "whisper-1"
openai.api_key = os.environ["OPENAI_API_KEY"]


def apply_whisper(filepath: str, mode: str) -> str:

    if mode not in ("translate", "transcribe"):
        raise ValueError(f"Invalid mode: {mode}")

    prompt = "Hello, this is a properly structured message. GPT, ChatGPT."
    
    with open(filepath, "rb") as audio_file:
        if mode == "translate":
            response = openai.Audio.translate(WHISPER_MODEL, audio_file, prompt=prompt)
        elif mode == "transcribe":
            response = openai.Audio.transcribe(WHISPER_MODEL, audio_file, prompt=prompt)

    transcript = response["text"]
    return transcript


def chatgpt(prompt: str, messages = None) -> str:
    """
    Uses the chat endpoint of the GPT-3 API to generate a response to a prompt.
    :param prompt:
    :return:
    """
    
    if messages == None or len(messages) == 0:
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant who responds in short answers."},
        ]
        
    messages.append({"role": "user", "content": prompt})
    
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages
    )
    
    messages.append(dict(response['choices'][0]['message']))
    return messages

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


def summarize(text):
    prompt = """
    You are a personal secretary, You will be fed a list of emails and you are to summarize them, extract key phrases, 
    determine their priority, and analyze their sentiment.
    
    I don't have time to read them all. So, I just want you to give me what's important in the email, a summary of it, 
    the key phrases, its priority (High, Medium, Low), and the overall sentiment (Positive, Negative, Neutral).
    
    If it's not an important email, just say where it is from and say nothing is important in it.

    It will be in the following form:
    From: {mail@example.com}
    Subject: {EXAMPLE SUBJECT}
    Date: {2021-01-01}
    Text: {EXAMPLE TEXT}.
    
    You should respond in the following format:
    FROM: {mail@example.com}
    Summary: {SUBJECT, TEXT} {text}
    Key Phrases: {key phrases}
    Priority: {priority}
    Sentiment: {sentiment}
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]

    response = None
    try:
        print("Calling OpenAI GPT API...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.9,
        )
    except Exception as e:
        print("Something went wrong...")

    return response.choices[0].message.content


def text_to_speech(text: str) -> None:
    try:
        print("Calling OpenAI TTS API")
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )
    except Exception as e:
        print("Something went wrong with tts...")
        return

    output_dir = f"{os.getcwd()}/dump"
    output_file = os.path.join(output_dir, f"output_{os.urandom(4).hex()}.mp3")
    response.stream_to_file(output_file)

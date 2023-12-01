import openai
from pathlib import Path
from elevenlabs import generate, play, set_api_key
from config import API_KEY, EL_API_KEY, CHATBOT

set_api_key(EL_API_KEY)


def text_to_speech(text, voice_id):
    audio = generate(
        text=text,
        voice=voice_id,
        model='eleven_multilingual_v1'
    )

    play(audio)


def transcribe(audio, input_language):
    audio_file = Path(audio)
    audio_file = audio_file.rename(audio_file.with_suffix('.wav'))

    with open(audio_file, "rb") as file:
        openai.api_key = API_KEY
        result = openai.Audio.transcribe("whisper-1", file, language=CHATBOT[input_language]['lang'])
    return result['text']

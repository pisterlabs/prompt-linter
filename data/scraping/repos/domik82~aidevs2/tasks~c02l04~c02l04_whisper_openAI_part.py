import os
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv, find_dotenv
from icecream import ic
from common.logger_setup import configure_logger
import whisper

import openai  # for making OpenAI API calls


# 'please return transcription of this file: https://zadania.aidevs.pl/data/mateusz.mp3'
# 'hint': 'use WHISPER model - https://platform.openai.com/docs/guides/speech-to-text'

def download(url, file_name):
    if urlparse(url).scheme in ('http', 'https'):
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, "wb") as file:
                file.write(response.content)
    else:
        raise Exception(f"Failed to download file from {url}")


# define a wrapper function for seeing how prompts affect transcriptions
def openAI_transcribe(audio_filepath, prompt: str) -> str:
    """Given a prompt, transcribe the audio file."""

    transcript = openai.Audio.transcribe(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
    )
    return transcript["text"]


# this has worked but I can't get it to work on second laptop - seems that there is some lib conflict
def create_transcription_using_local(mp3_file, log):
    log.info(f"mp3_file:{mp3_file}")
    try:
        if os.path.isfile(mp3_file) and mp3_file.endswith('.mp3'):

            # https://github.com/openai/whisper#available-models-and-languages
            model = whisper.load_model("tiny")  # best for testing pure code (not data extraction)
            # model = whisper.load_model("base")

            # bare minimum to get relatively good translation -> small
            # model = whisper.load_model("small")

            # 100x slower on my old laptop without good GPU
            # model = whisper.load_model("large-v2")

            # ffmpeg is mandatory
            transcript = model.transcribe(mp3_file, verbose=True)
            log.info(f"transcript:{transcript}")
            return transcript
        else:
            raise Exception(f"{mp3_file} is not a valid .mp3 file")
    except Exception as e:
        log.exception(f"Exception: {e}")


def get_file_name(url):
    return urlparse(url).path.split("/")[-1]


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    log = configure_logger("whisper_openAI")
    try:
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        url = 'https://zadania.aidevs.pl/data/mateusz.mp3'

        downloaded_file = get_file_name(url)
        if not os.path.exists(downloaded_file):
            download(url, downloaded_file)

        transcribed = openAI_transcribe(downloaded_file, prompt="As Polish would say:")
        ic(transcribed)

        # transcribed2 = create_transcription_using_local(downloaded_file, log)
        # ic(f'transcribed2: {transcribed2["text"]}')

    except Exception as e:
        log.exception(f"Exception: {e}")

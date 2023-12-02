import os
import tempfile
import warnings

import dotenv
import openai
from gradio.processing_utils import audio_to_file

dotenv.load_dotenv()

OPENAI_WHISPER_API_TYPE = os.getenv("OPENAI_WHISPER_API_TYPE")
OPENAI_WHISPER_API_BASE = os.getenv("OPENAI_WHISPER_API_BASE")
OPENAI_WHISPER_API_KEY = os.getenv("OPENAI_WHISPER_API_KEY")
OPENAI_WHISPER_API_VERSION = os.getenv("OPENAI_WHISPER_API_VERSION")
OPENAI_WHISPER_MODEL_NAME = os.getenv("OPENAI_WHISPER_MODEL_NAME")


def transcribe_wav_file(wav_file, prompt):
    assert OPENAI_WHISPER_API_TYPE in [None, "azure", "openai"]
    kwargs = {
        "file": wav_file,
        "api_type": OPENAI_WHISPER_API_TYPE,
        "api_base": OPENAI_WHISPER_API_BASE,
        "api_key": OPENAI_WHISPER_API_KEY,
        "api_version": OPENAI_WHISPER_API_VERSION,
    }
    if OPENAI_WHISPER_API_TYPE == "azure":
        kwargs["model"] = OPENAI_WHISPER_MODEL_NAME
        kwargs["deployment_id"] = OPENAI_WHISPER_MODEL_NAME
    else:  # openai
        kwargs["model"] = OPENAI_WHISPER_MODEL_NAME

    transcript = openai.Audio.transcribe(**kwargs, prompt=prompt)

    return transcript.text


def transcribe_audio_data(sr, data, prompt=None):
    with tempfile.NamedTemporaryFile(suffix=".wav") as file:
        with warnings.catch_warnings():
            # ignore warnings of converting int32 to int16
            warnings.simplefilter("ignore")
            audio_to_file(sr, data, file.name, format="wav")
        with open(file.name, "rb") as f:
            transcript = transcribe_wav_file(f, prompt)
        return transcript


if __name__ == "__main__":
    file = open("/path/to/sample.wav", "rb")
    print(transcribe_wav_file(file))

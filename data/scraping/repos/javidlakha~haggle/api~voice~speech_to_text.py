import math
import pathlib
from io import BytesIO
from tempfile import TemporaryDirectory

import openai
from pydub import AudioSegment

from api.settings import OPENAI_API_KEY

BASE = pathlib.Path(__file__).parent.parent.parent.resolve()


def speech_to_text(recording: bytes) -> str:
    """Converts an uploaded recording to text"""
    recording = AudioSegment.from_file(BytesIO(recording))
    transcript = ""
    with TemporaryDirectory() as temp_dir:
        for start_time in range(0, math.ceil(recording.duration_seconds), 60):
            snippet_path = f"{temp_dir}/snippet.mp3"
            snippet = recording[start_time * 1000 : (start_time + 60) * 1000]
            snippet.export(snippet_path)
            with open(snippet_path, "rb") as g:
                segment_transcript = openai.Audio.transcribe("whisper-1", g, api_key=OPENAI_API_KEY)["text"]
                transcript += segment_transcript + " "
        
    return transcript

import os
import pathlib
import math
from pathlib import Path

import openai
from pydub import AudioSegment
from moviepy.editor import AudioFileClip

openai.api_key = os.environ["OPENAI_API_KEY"]


def convert_to_mp3(input_file, output_file):
    clip = AudioFileClip(input_file)
    clip.write_audiofile(output_file)
    clip.close()


def split_mp3(input_file: str, output_prefix: str, folder_path: pathlib.Path, duration: int = 300000):
    folder_path = Path(folder_path)
    audio = AudioSegment.from_mp3(input_file)
    total_duration = len(audio)
    num_parts = math.ceil(total_duration / duration)

    for i in range(num_parts):
        start_time = i * duration
        end_time = min((i + 1) * duration, total_duration)
        part = audio[start_time:end_time]
        path_template = str(folder_path / f"{output_prefix}_") + "{}.mp3"
        output_file = path_template.format(i + 1)
        part.export(output_file, format="mp3")
        print(f"Exported {output_file}")

    return num_parts, path_template


def transcribe_mp3_group(file_template: str, num_parts: int) -> str:
    transcripts = []
    for i in range(num_parts):
        path = str(file_template.format(i + 1))
        with open(path, "rb") as audio_file:
            whisper_obj = openai.Audio.transcribe("whisper-1", audio_file)
            transcripts.append(whisper_obj.text)
    full_text = "\n\n".join(transcripts)
    return full_text

from openai import OpenAI
from config import Config
import os

client = OpenAI(api_key=Config.openai_api_key)


def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if not file.startswith('part_output_'):
                continue
            yield file


for filename in get_files(os.path.join(os.getcwd(), 'tmp')):
    path = os.path.join(os.getcwd(), 'tmp', filename)

    transcript_filename = f"trans_{filename}.txt"
    transcript_filename_path = os.path.join(os.getcwd(), 'tmp', transcript_filename)
    if os.path.exists(transcript_filename_path):
        print(f"{filename} already done")
        continue

    f = open(transcript_filename_path, "w+")

    print(f"{filename} started")
    audio_file = open(path, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text",
    )

    f.write(transcript)
    f.close()

    print(f"{filename} is done")

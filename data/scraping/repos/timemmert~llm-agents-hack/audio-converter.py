import os

import openai

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
# Iterate over the data/db/audio directory
# For each file, transcribe it and save the transcription to a file in the data/db/transcriptions directory
path_of_this_python_folder = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path_of_this_python_folder, f"data/db/audio")

files = os.listdir(path)
for file in files:
    name = file.split(".")[0]
    audio_file = open(os.path.join(path, file), "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    path = os.path.join(path_of_this_python_folder, f"../data/db/people/{name}.txt")
    with open(path, "w") as file:
        file.write(transcript)
    print(f"Transcribed {file}")

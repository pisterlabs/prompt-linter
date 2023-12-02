"""
Class that compiles the necessary speech recognition data and automates P2FA's align.py to
generate 500+ batches of data sets.

We use OpenAI's Whisper model to transcribe an audio file and create a transcription file
that is stored to P2FA's input directory. We then use both the audio and transcription file
as a singular input pair to generate its TextGrid file.

Each input pair should have the same exact name, specifically participant#_p14_trial#.txt;
Must be labeled p14 to signify cases detailing contrastive stress production collected
by Language Acquisition and Processing Lab.
"""
import os
from pathlib import Path
import openai

# Import OpenAI API Key.
openai.api_key = ''

# Obtain local directory containing audio files.
audio_dir = Path('/Users/jasmeanfernando/PycharmProjects/BatchP2FA/p2fa/input_wav')

# Obtain local directory to store transcription files.
transcription_dir = Path('/Users/jasmeanfernando/PycharmProjects/BatchP2FA/p2fa/input_txt')

# List of tuples containing .wav, .txt, and .TextGrid files.
arglist = []

for audio_file in os.listdir(audio_dir):
    # Base Case: Check if .wav file.
    if audio_file.endswith(".wav"):
        # Obtain audio_file name.
        audio_file_name = audio_file.split(".")[0]

        # Open audio.
        wav_path = audio_dir.joinpath(audio_file)
        audio = open(wav_path, "rb")

        # Transcribe audio.
        transcription = openai.Audio.transcribe("whisper-1", audio)
        print(transcription["text"])

        # Initialize .txt file.
        transcription_file = audio_file_name + '.txt'
        txt_path = transcription_dir.joinpath(transcription_file)

        # Base Case: Check if .txt file /already/ exists.
        if txt_path.is_file():
            print("File already exists, cannot re-write.")
        else:
            # Write and store .txt file.
            with open(txt_path, "w") as file:
                file.write(str(transcription["text"]))
                print("Creating...", transcription_file)

        # Initialize .TextGrid file.
        textgrid_file = audio_file_name + '.TextGrid'

        # Append .wav, .txt, and .TextGrid files.
        arglist.append([audio_file, transcription_file, textgrid_file])

i = 0
for vars in arglist:
    print('Creating...' + arglist[i][2])
    os.system('python3 align.py input_wav/' + arglist[i][0] + ' input_txt/' + arglist[i][1] + ' output_textgrid/' + arglist[i][2])
    i = i + 1
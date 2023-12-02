import openai
import __api
import os

openai.api_key = __api.api

onlyfiles = [f for f in os.listdir('/Users/coding/Documents/vs/FluentFriend/model/database/db_input')]

def S2T(audio):
    with open(audio, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language="en"
        )
    return transcript

def process_audio_files(directory,dataset_directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)

            basename = os.path.splitext(filename)[0]

            speech_to_text_output = S2T(filepath)

            with open(os.path.join(dataset_directory, basename + ".txt"), "a") as f:
                f.writelines(speech_to_text_output)

def runner_1():
    try:
        input_directory = "/Users/coding/Documents/vs/FluentFriend/model/database/db_input/"
        dataset_directory = "/Users/coding/Documents/vs/FluentFriend/model/dataset/input/"

        process_audio_files(input_directory,dataset_directory)
            
    except:
        pass

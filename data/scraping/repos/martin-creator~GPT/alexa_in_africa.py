# # How my alexa works in Africa
# # The user asks a question using their microphone. We will use Python SpeechRecognition⁴⁶.
# # • OpenAI Whisper automatically transcribes the question into a text.
# # • The text is passed to OpenAI GPT completions endpoints
# # • The OpenAI API returns an answer
# # • The answer is saved to an mp3 file and passed to Google Text to Speech (gTTS) to create a
# # voice response.



# # Recording the audio
# # We will use the Python SpeechRecognition library to record the audio

# from pydub import AudioSegment
# from pydub.playback import play
# import speech_recognition as sr
# import whisper
# import queue
# import os
# import threading
# import torch
# import numpy as np
# import re
# from gtts import gTTS
# import openai
# import click

# import os
# import openai
# import click

# # develop a command-line tool that can assist us with Linux commands through conversation.
# # Click documentation: https://click.palletsprojects.com/en/8.1.x/


# def init_api():
#     ''' Load API key from .env file'''
#     with open(".env") as env:
#         for line in env:
#             key, value = line.strip().split("=")
#             os.environ[key] = value

#     openai.api_key = os.environ["API_KEY"]
#     openai.organization = os.environ["ORG_ID"]



# def record_audio(audio_queue, energy, pause, dynamic_energy):
#     """
#         • audio_queue: a queue object where the recorded audio will be saved.
#         • energy: an initial energy threshold for detecting speech.
#         • pause: a pause threshold for detecting when speech has ended.
#         • dynamic_energy: a Boolean indicating whether the energy threshold should be adjusted
#           dynamically based on the surrounding environment
#     """
#     # load the speec recognizer and set the intial energy threshold snd pause threshold
#     r = sr.Recognizer()
#     r.energy_threshold = energy
#     r.pause_threshold = pause
#     r.dynamic_energy_threshold = dynamic_energy


#     # use the microphone as source for input
#     with sr.Microphone(sample_rate=16000) as source:
#         print("Listening...")
#         i = 0

#         while True:
#             # get and save audio to wav file
#             audio = r.listen(source)

#         # Using: https://github.com/openai/whisper/blob/9f7016 532bd8c21d/whisper/audio.py
#             torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
#             # we use 32768.0 becasue the audio is 16 bit and we want to normalize it to -1 to 1
#             audio_data = torch_audio
#             audio_queue.put_nowait(audio_data)
#             i += 1


# # Transcribing the audio
# # OpenAI Whisper automatically transcribes the question into a text.

# def transcribe_forever(audio_queue, result_queue, audio_model, english, wake_word, verbose):
#     """
#     audio_queue, which contains the audio data to be transcribed
#     result_queue, which is used to store the transcribed text
#     """
#     while True:
#         audio_data = audio_queue.get()
#         if english:
#             result = audio_model.transcribe(audio_data, language="english", verbose=verbose)
#         else:
#             result = audio_model.transcribe(audio_data)

#         predicted_text = result["text"]

#         if predicted_text.strip().lower().startswith(wake_word.strip().lower()):
#             pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
#             predicted_text = pattern.sub("", predicted_text).strip()
#             punc = ''' !()-[]{};:'"\,<>./?@#$%^&*_~ '''
#             predicted_text.translate({ord(char): None for char in punc})
#             if verbose:
#                 print("You said the wake word ... Processing {}".format(predicted_text))
#             result_queue.put_nowait(predicted_text)
#         else:
#             if verbose:
#                 print("You did not say the wake word .. Ignoring")


# # Generating the answer

# def reply(result_queue, verbose):
#     while True:
#         question = result_queue.get()
#         # We use the following format for the prompt: "Q: <question>?\nA:"
#         prompt = "Q: {}?\nA:".format(question)

#         data = openai.Completion.create(
#             model="text-davinci-002",
#             prompt=prompt,
#             temperature=0.5,
#             max_tokens=100,
#             n=1,
#             stop=["\n"]
#         )

#         # answer = data["choices"][0]["text"]
#         # mp3_obj = gTTS(text=answer, lang="en", slow=False)
#         # mp3_obj.save("reply.mp3")
#         # reply_audio = AudioSegment.from_mp3("reply.mp3")
#         # play(reply_audio)
#         # os.remove("reply.mp3")

#         # We catch the execption in case the API returns an empty response
#         try:
#             answer = data["choices"][0]["text"]
#             mp3_obj = gTTS(text=answer, lang="en", slow=False)
#         except Exception as e:
#             choices = ["I'm sorry, I don't know the answer to that", "I'm not sure I understand", "I'm not sure I can answer that", "Please repeat the question in a different way"]
#             mp3_obj = gTTS(text=choices[np.random.randint(0, len(choices))], lang="en", slow=False)

#             if verbose:
#                 print("Error: {}".format(e))
            
#             # In both cases we save the audio to a file and play it
#             mp3_obj.save("reply.mp3")
#             reply_audio = AudioSegment.from_mp3("reply.mp3")
#             play(reply_audio)


# # Main function 
# # read from arguments

# @click.command()
# @click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "medium", "large"]))
# @click.option("--english", default=False, help="Whether to use English model", is_flag=True, type=bool)
# @click.option("--energy", default=300, help="Energy level for the mic to detact", type=int)
# @click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
# @click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic energy", type=bool)
# @click.option("--wake_word", default="hey computer", help="Wake word to listen for", type=str)
# @click.option("--verbose", default=False, is_flag=True, help="Whether to print the verbose output", type=bool)


# def main(model, english, energy, pause, dynamic_energy, wake_word, verbose):
#     # there are no english models for large
#     if model != "large" and english:
#         model = model + ".en"
    
#     audio_model = whisper.load_model(model)
#     audio_queue = queue.Queue()
#     result_queue = queue.Queue()
#     threading.Thread(target=record_audio, args=(audio_queue, energy, pause, dynamic_energy)).start()
#     threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, audio_model, english, wake_word, verbose)).start()
#     threading.Thread(target=reply, args=(result_queue,)).start()


#     while True:
#         print(result_queue.get())


# # Main entry point
# init_api()
# main()



from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import whisper
import queue
import os
import threading
import torch
import numpy as np
import re
from gtts import gTTS
import openai
import click

import os
import openai
import click

# Load API key from .env file
def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ["API_KEY"]
    openai.organization = os.environ["ORG_ID"]

# Record the audio
def record_audio(audio_queue, energy, pause, dynamic_energy):
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Listening...")
        while True:
            audio = r.listen(source)
            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_data = torch_audio
            audio_queue.put(audio_data)

# Transcribe the audio
def transcribe_forever(audio_queue, result_queue, audio_model, english, wake_word, verbose):
    while True:
        audio_data = audio_queue.get()
        if english:
            result = audio_model.transcribe(audio_data, language="english", verbose=verbose)
        else:
            result = audio_model.transcribe(audio_data)

        predicted_text = result["text"]

        if predicted_text.strip().lower().startswith(wake_word.strip().lower()):
            pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
            predicted_text = pattern.sub("", predicted_text).strip()
            punc = ''' !()-[]{};:'"\,<>./?@#$%^&*_~ '''
            predicted_text.translate({ord(char): None for char in punc})
            if verbose:
                print("You said the wake word... Processing: {}".format(predicted_text))
            result_queue.put(predicted_text)
        else:
            if verbose:
                print("You did not say the wake word... Ignoring")

# Generate the answer
def reply(result_queue, verbose):
    while True:
        question = result_queue.get()
        prompt = "Q: {}?\nA:".format(question)

        data = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
            n=1,
            stop=["\n"]
        )

        try:
            answer = data["choices"][0]["text"]
            mp3_obj = gTTS(text=answer, lang="en", slow=False)
        except Exception as e:
            choices = ["I'm sorry, I don't know the answer to that", "I'm not sure I understand", "I'm not sure I can answer that", "Please repeat the question in a different way"]
            mp3_obj = gTTS(text=choices[np.random.randint(0, len(choices))], lang="en", slow=False)

            if verbose:
                print("Error: {}".format(e))
            
        mp3_obj.save("reply.mp3")
        reply_audio = AudioSegment.from_mp3("reply.mp3")
        play(reply_audio)
        os.remove("reply.mp3")

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "medium", "large"]))
@click.option("--english", default=False, help="Whether to use English model", is_flag=True, type=bool)
@click.option("--energy", default=300, help="Energy level for the mic to detect", type=int)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--wake_word", default="hey computer", help="Wake word to listen for", type=str)
@click.option("--verbose", default=False, is_flag=True, help="Whether to print the verbose output", type=bool)
def main(model, english, energy, pause, dynamic_energy, wake_word, verbose):
    if model != "large" and english:
        model = model + ".en"
    
    audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()

    threading.Thread(target=record_audio, args=(audio_queue, energy, pause, dynamic_energy)).start()
    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, audio_model, english, wake_word, verbose)).start()
    threading.Thread(target=reply, args=(result_queue,)).start()

if __name__ == "__main__":
    init_api()
    main()

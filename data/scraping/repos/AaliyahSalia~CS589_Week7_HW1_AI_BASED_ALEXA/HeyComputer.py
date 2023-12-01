#Alexa.py
from pydub import AudioSegment
import simpleaudio as sa
import io
import wave
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
import time
                              
def init_api():
    try:
        with open("/Users/aaliyahsalia/Desktop/SFBU/6thTrimester/CS589/Week7_HW1/.env.", "r") as env:  # Make sure this path is correct
            for line in env:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value.replace('"', '')  # Removing quotes if present
    except IOError:
        print("Could not open .env file.")
        exit(1)
    
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        print("No API key provided.")
        exit(1)

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "small", "medium", "large"]))
@click.option("--english", default=False, help="Whether to use the English model", is_flag=True, type=bool)
@click.option("--energy", default=300, help="Energy level for the mic to detect", type=int)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--wake_word", default="hey computer", help="Wake word to listen for", type=str)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)

def main(model, english, energy, pause, dynamic_energy, wake_word, verbose):
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    
    threading.Thread(target=record_audio, args=(audio_queue, energy, pause, dynamic_energy,)).start()
    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, audio_model, english, wake_word, verbose,)).start()
    threading.Thread(target=reply, args=(result_queue, verbose,)).start()

    while True:
        print(result_queue.get())

def record_audio(audio_queue, energy, pause, dynamic_energy):
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Listening...")
        i = 0
        while True:
            audio = r.listen(source)
            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_data = torch_audio
            audio_queue.put_nowait(audio_data)
            i += 1

def transcribe_forever(audio_queue, result_queue, audio_model, english, wake_word, verbose):
    while True:
        audio_data = audio_queue.get()
        # Start of transcription
        print(f"Starting transcription at {time.strftime('%X')}")
        if english:
            result = audio_model.transcribe(audio_data, language='english')
        else:
            result = audio_model.transcribe(audio_data)
        
        # End of transcription
        print(f"Finished transcription at {time.strftime('%X')} - Result: {result['text'][:50]}...")
        
        predicted_text = result["text"]
        
        if predicted_text.strip().lower().startswith(wake_word.strip().lower()):
            pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
            predicted_text = pattern.sub("", predicted_text).strip()
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            predicted_text = predicted_text.translate({ord(i): None for i in punc})
            if verbose:
                print("You said the wake word.. Processing {}...".format(predicted_text))

            result_queue.put_nowait(predicted_text)
        else:
            if verbose:
                print("You did not say the wake word.. Ignoring")

def play_audio(audio_segment):
    # Convert the AudioSegment to a WAV byte stream
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format='wav')
    wav_io.seek(0)
    wav_data = wav_io.read()

    # Play the audio using simpleaudio
    wave_obj = sa.WaveObject.from_wave_file(io.BytesIO(wav_data))
    play_obj = wave_obj.play()
    play_obj.wait_done()

def reply(result_queue, verbose):
    while True:
        question = result_queue.get()
        if verbose:
            print(f"[{time.strftime('%X')}] Received question: {question}")

        # Formulate the prompt with additional context or instructions
        prompt = (
            "The following is a question from a user who needs a detailed and informative answer.\n"
            "User's question: {}\n"
            "AI's detailed response:".format(question)
        )
        try:
            if verbose:
                print(f"[{time.strftime('%X')}] Sending prompt to OpenAI API: {prompt}")
            response = openai.Completion.create(
                engine="text-davinci-003",  # Make sure to use the latest engine
                prompt=prompt,
                temperature=0.5,  # A lower temperature for more concise and on-topic responses
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            answer = response.choices[0].text.strip()
            if not answer:
                raise ValueError("Received empty response from OpenAI API.")
        except Exception as e:
            if verbose:
                print(f"[{time.strftime('%X')}] Error: {e}")
            answer = "I'm sorry, I didn't understand that. Could you provide more details or context?"

        # Generate the spoken response
        if verbose:
            print(f"[{time.strftime('%X')}] Generating audio for the response.")
        mp3_obj = gTTS(text=answer, lang="en", slow=False)
        mp3_fp = io.BytesIO()
        mp3_obj.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Convert MP3 data to WAV and play the response
        sound = AudioSegment.from_file(mp3_fp, format="mp3")
        wav_fp = io.BytesIO()
        sound.export(wav_fp, format="wav")
        wav_fp.seek(0)
        if verbose:
            print(f"[{time.strftime('%X')}] Playing the response.")
        play_obj = sa.WaveObject.from_wave_file(wav_fp).play()
        play_obj.wait_done()

        
init_api()
main()



        
    


import pyaudio
import wave
import audioop
import math
import openai
import pygame
from google.cloud import texttospeech
import time
import speech_recognition as sr
import simpleaudio as sa
import numpy as np
from colorama import Fore, Style
from langdetect import detect
import asyncio
import os
CHUNK = 1024  # number of audio frames per buffer
FORMAT = pyaudio.paInt16  # audio format (16-bit signed integer)
CHANNELS = 1  # number of audio channels (mono)
RATE = 44000  # sample rate (Hz)
SILENCE_LIMIT = 0.5  # seconds of silence needed to stop recording
THRESHOLD = 40
interupt = False


def beep():
    """Plays a beep sound using simpleaudio."""
    frequency = 1000  # frequency of the beep in Hz
    duration = 100  # duration of the beep in milliseconds
    t = np.linspace(0, duration/1000, int(duration*44100/1000), False)
    beep_wave = np.sin(2*np.pi*frequency*t)
    beep_wave *= 32767 / np.max(np.abs(beep_wave))
    beep_wave = beep_wave.astype(np.int16)
    play_obj = sa.play_buffer(beep_wave, 1, 2, 44100)
    play_obj.wait_done()
    play_obj.stop()


def text_to_speech(text, output_path):
    """Synthesizes text to speech using Google Text-to-Speech API.

    Args:
        text (str): The text to synthesize.
        output_path (str): Path to save the output audio file.

    Raises:
        Exception: If there's an error during speech synthesis.
    """
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    try:
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config)
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
    except Exception as e:
        raise Exception(f"Google Text-to-Speech API error: {e}")


def speech_to_text():
    """
    Synthesizes text to speech using the OpenAI Audio API.

    Args:
        None.

    Returns:
        The resulting text from the transcription process as a string.

    Raises:
        An exception if there are any errors during the transcription process.
    """
    try:
        with open("MyVoice.wav", "rb") as audio_file:
            result = openai.Audio.transcribe(
                "whisper-1", audio_file)['text']
        return result
    except Exception as e:
        raise Exception(f"OpenAI Whisper API error: {e}")


def saveRecordedVoice(frames, audio):
    """
    Saves recorded audio frames to disk as a WAV file named "MyVoice.wav".

    Args:
        frames: A list of audio frames to be saved.
        audio: An instance of the audio module.

    Returns:
        None.

    Raises:
        An exception if there are any errors during the file writing process.
    """
    try:
        filename = "MyVoice.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        frames = []
    except Exception as e:
        raise Exception(f"Error on saving recorded voice: {e}")


async def gpt_request(result):
    """
    Sends a user's text input to the OpenAI GPT-3.5 model for response.

    Args:
        result: A string representing the user's input.

    Returns:
        A string representing the GPT-3.5 model's response to the user's input.

    Raises:
        An exception if there are any errors during the request process.
    """
    try:
        loop = asyncio.get_running_loop()
        response = await asyncio.wait_for(loop.run_in_executor(None, lambda: openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an english teacher for beginners. ask me questions. correct my answer. answer with maximum 50 words"},
                {"role": "user", "content": result}
            ]
        )), timeout=10)
        if response:
            return response['choices'][0]['message']['content']
        else:
            return None
    except asyncio.TimeoutError:
        print("Request timed out after 10 seconds")
    except Exception as e:
        raise Exception(f"OpenAI GPT API error: {e}")


def playResponse():
    """
    Plays an audio response from the GPT-3.5 model using the Pygame mixer module.

    Args:
        None.

    Returns:
        None.

    Raises:
        An exception if there are any errors during the playback process.
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("GptVoice.wav")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        raise Exception(f"Audio playback error: {e}")


async def listen():
    global interupt
    frames = []
    recording = False
    silence_counter = 0
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    while True:
        data = stream.read(CHUNK)
        rms = audioop.rms(data, 2)
        decibel = 20 * math.log10(rms)

        if decibel > THRESHOLD:
            if not recording:

                recording = True
            frames.append(data)
            silence_counter = 0
        else:
            if recording:
                silence_counter += 1

            if silence_counter > SILENCE_LIMIT * (RATE / CHUNK):
                if recording:
                    recording = False
                    if len(frames) < 5:
                        break
                    beep()
                    saveRecordedVoice(frames, audio)
                    result = speech_to_text()
                    print("Recording stopped...")
                    print("------------------------------------------")

                    print(Fore.GREEN + "=> Recorded Text: " +
                          Style.RESET_ALL + result)

                    response = await gpt_request(result)
                    if (response == None):
                        print(
                            Fore.RED + "Error on getting GPT Response. Please try again..." + Style.RESET_ALL)
                        print("==========================================")
                        print("\n")
                        break
                    print(Fore.GREEN + "=> GPT Response: " +
                          Style.RESET_ALL + response)
                    print("==========================================")
                    print("\n")
                    text_to_speech(response, 'GptVoice.wav')
                    playResponse()
                    silence_counter = 0
                    time.sleep(1)
                    beep()
                    print(Fore.MAGENTA + "What is your question? Ask GPT..." +
                          Style.RESET_ALL)

                    break


async def main():
    openai.api_key = os.getenv('openai_key')
    recognizer = sr.Recognizer()
    global interupt
    with sr.Microphone(sample_rate=44000) as source:
        while True:
            interupt = False
            print(Fore.YELLOW + 'listening...' +
                  Style.RESET_ALL, end='\r', flush=True)
            audio = recognizer.listen(source)
            rms = audioop.rms(audio.frame_data, audio.sample_width)
            if rms > THRESHOLD:
                try:
                    text = recognizer.recognize_google(audio)
                except:
                    # print("Error getting activation text, re-listening...")
                    continue
                if "hello" in text:
                    beep()
                    print("\n")
                    print("==========================================")
                    print("Recording started...")
                    print(Fore.MAGENTA + "What is your question? Ask GPT..." +
                          Style.RESET_ALL)
                    while not interupt:
                        try:
                            # calling the listen() function using await
                            await listen()
                        except Exception as e:
                            print(f"Error on listening process: {e}")
                            print("re-listening...")

if __name__ == '__main__':
    asyncio.run(main())

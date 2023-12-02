import array
import json
import os
import subprocess
import threading
import time

import azure.cognitiveservices.speech as speechsdk
import numpy
import openai

# from database import save_transcription
# from dotenv import load_dotenv

# _ = load_dotenv()

subscription_key = "598d3431577d4431bf8ff02c77eb27b7"
service_region = "eastasia"
custom_endpoint = "1bd6d651-d0d9-41bf-b737-f4463cc2eca7"
openai.api_key = os.getenv("OPENAI_API_KEY")
gpt_model = os.getenv("GPT_MODEL")
language_code = "zh-HK"
# subscription_key = os.getenv("SPEECH_KEY")
# service_region = os.getenv("SERVICE_REGION")
# custom_endpoint = os.getenv("ENDPOINT_ID")
# openai.api_key = os.getenv("OPENAI_API_KEY")
# gpt_model = os.getenv("GPT_MODEL")
# language_code = os.getenv("LANGUAGE_CODE")


def generate_summary(transcriptions, language_code):
    # Join all transcriptions into a single string
    transcription_text = " ".join(transcriptions)

    # Generate summary using OpenAI GPT-3.5 model
    prompt = f"[{language_code}] Please summarize the following transcriptions:\n{transcription_text}\nSummary:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    summarized_text = response.choices[0].text.strip()
    return summarized_text


def transcribe_microphone():
    language_code = "zh-HK"
    speech_config = speechsdk.SpeechConfig(
        subscription=subscription_key, region=service_region
    )
    speech_config.endpoint_id = custom_endpoint
    speech_config.speech_recognition_language = language_code
    # audio_config = speechsdk.AudioConfig(filename="output.wav")
    audio_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    done = False
    transcription = ""

    def stop_cb(evt):
        nonlocal done
        done = True

    def speech_recognized(evt):
        nonlocal transcription
        result = evt.result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcription += result.text
            print("trans: {}".format(result.text))

    speech_recognizer.recognized.connect(speech_recognized)
    speech_recognizer.session_stopped.connect(stop_cb)

    speech_recognizer.start_continuous_recognition()

    # Keep transcribing until manually stopped
    while not done:
        pass

    speech_recognizer.stop_continuous_recognition()
    print("transcription was done.")

    # Save transcription to SQLite database
    # save_transcription(transcription, language_code)
    print("transcription was saved.", transcription)
    return transcription


def send_message(summary: str, initial_prompt) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": initial_prompt},
            {"role": "user", "content": summary},
        ],
        temperature=0,
    )
    return response.choices[0].message


class AzureEngine:
    def __init__(self, language_code):
        # Configure speech recognition
        speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=service_region
        )
        speech_config.endpoint_id = custom_endpoint
        speech_config.speech_recognition_language = language_code
        speech_config.enable_streaming = True

        self.audio_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.audio_stream)

        # Create a speech recognizer with streaming support
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        # Connect the callbacks to the speech recognizer events
        self.speech_recognizer.recognizing.connect(self.process_audio_chunk)
        self.speech_recognizer.recognized.connect(self.process_audio_chunk)
        self.speech_recognizer.canceled.connect(self.process_audio_chunk)
        self.speech_recognizer.start_continuous_recognition_async()

    # Set up the callbacks for handling streaming data
    def process_audio_chunk(self, args):
        result = args.result
        if result.reason == speechsdk.ResultReason.RecognizingSpeech:
            print("Recognizing:", result.text)
        elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(result.text))
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech recognition canceled. Reason:", cancellation_details.reason)
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details:", cancellation_details.error_details)

    def process_audio_stream(self, event):
        audio_stream = event.input_stream
        chunk_size = 3200  # Adjust the chunk size as per your requirements
        while not audio_stream.is_closed:
            audio_data = audio_stream.read(chunk_size)
            self.speech_recognizer.push_audio_buffer(audio_data)

    def get_transcription(self, frame):
        # get float data
        # self.speech_recognizer.stop_continuous_recognition()
        # print("transcription was done.")
        self.audio_stream.write(frame)

        stt_res = {"partial": ""}
        # check if we will get transcription
        return json.dumps(stt_res)


def transcribe(wav_file):
    sample_rate = 16000
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-i",
            wav_file,
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
    )

    # whisper = WhisperEngine("en", model_path)
    azuer = AzureEngine("zh-HK")

    while True:
        data = process.stdout.read(1024)
        if len(data) == 0:
            break
        # print("read data")
        time.sleep(1024 / 16000.0)
        azuer.get_transcription(data)

    # When you're done feeding audio data, close the audio stream
    azuer.audio_stream.close()

    # Stop speech recognition and clean up
    azuer.speech_recognizer.stop_continuous_recognition()
    # azuer.speech_recognizer.close()


if __name__ == "__main__":
    wav_file = "out.mp3"
    transcribe(wav_file)

from dotenv import load_dotenv
import time
import os
from sys import platform

import openai
import speech_recognition as sr
import whisper
import pyaudio
import wave

from faster_whisper import WhisperModel

from ASR.whisperASR import Wisp
from ASR.googleASR import GScribe
from ASR.assemblyASR import Ass
from util import Util

# Load .env file
load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

class RTT:
    def __init__(self, engine=None, microphone=None):
        """ If no engine is specified, the default is Google's Speech Recognition API """
        self.engine = engine
        self.microphone = microphone if microphone else sr.Microphone()
        self.fp16 = True if platform == 'win32' else False

    def transcribe_audio_to_text(self, filename):
        """Transcribe audio to text. As of now, Google's Speech Recognition API is faster than Whisper"""
        if isinstance(self.engine, Wisp):
            text = self.engine.transcribe(filename)
        else:
            gscribe = GScribe()
            text = gscribe.transcribe(filename)

        # Print transcription
        if text:
            print(f"{'Wisp' if self.engine else 'Google'}: {text}")
            # print(text)

    def RTT_mic(self):
        """Records and transcribes mic audio to text"""
        try:
            # Record audio
            filename = 'RTT.wav'
            with self.microphone as source:
                recognizer = sr.Recognizer()
                recognizer.adjust_for_ambient_noise(source)
                # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
                recognizer.dynamic_energy_threshold = False
                source.energy_threshold = 300
                source.pause_threshold = 0
                audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                with open(filename, "wb") as f:
                    f.write(audio.get_wav_data())
                
            # Transcribe audio to text. As of now, Google's Speech Recognition API is faster than Whisper
            self.transcribe_audio_to_text(filename)
                    
        except Exception as e:
            print("[RTT_mic] An error occurred: {}".format(e))

    def RTT_system(self):
        """Records and transcribes system audio to text"""
        try:
            # Record audio
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100
            RECORD_SECONDS = 5
            WAVE_OUTPUT_FILENAME = "output.wav"

            p = pyaudio.PyAudio()

            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

            print("* recording")

            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            print("* done recording")

            stream.stop_stream()
            stream.close()
            p.terminate()

            with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))


            # Transcribe audio to text. As of now, Google's Speech Recognition API is faster than Whisper
            self.transcribe_audio_to_text(WAVE_OUTPUT_FILENAME)

            pass
        except Exception as e:
            print("[RTT_system] An error occurred: {}".format(e))

    def Assembly_mic(self):
        try:
            ass = Ass()
            ass.run()
        except:
            pass

def main():
    # Debugging: Print all microphone names
    # print(sr.Microphone.list_microphone_names())
    microphone = sr.Microphone(device_index=1, sample_rate=16000)  # Microphone device index

    # Load Whisper model
    print("\033[32mLoading Whisper Model...\033[37m")
    wisp = Wisp('medium', microphone=microphone)
    
    # Set engine = None if you want to use Google's Speech Recognition API
    rtt = RTT(engine=wisp, microphone=microphone)
    # rtt = RTT(engine=None, microphone=microphone)

    # Start Recording
    print("\033[32mRecording...\033[37m(Ctrl+C to Quit)\033[0m")

    # Record and Transcribe Audio until Ctrl+C is pressed
    while True:    
        try:
            rtt.RTT_mic()
        except (KeyboardInterrupt, SystemExit): break

if __name__ == "__main__":
    main()
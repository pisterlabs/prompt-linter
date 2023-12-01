import openai
import sounddevice as sd
from TTS.api import TTS
import numpy as np
import pyrubberband as pyrb
import queue, threading, time
from config import *
import torch
import soundfile as sf
from lipsync import Lipsync
from PyQt6.QtMultimedia import *
from PyQt6.QtCore import QUrl

import sys
from PyQt6.QtGui import QGuiApplication

from pygame import mixer

# openai.api_key = OPEN_AI_KEY
class Sound():
    # curr_playing = False
    audio = []
    def __init__(self, lip_queue=None, save_audio=True, save_text=True, volume=1):
        # x = torch.rand(5, 3)
        # print(x)
        # print(torch.cuda.is_available())
        self.lip_queue = lip_queue

        self.tts = TTS("tts_models/en/ljspeech/vits", gpu=False)
        # tts_models/en/vctk/vits"

        self.speak_queue = queue.Queue()

        self.client = openai.OpenAI(api_key=OPEN_AI_KEY)

        # audio stuff
        self.curr_playing = False
        self.delay = 1  #delay in secs before saying something new
        self.last_speak = time.time()

        self.audio = []
        self.volume = volume
        self.lipsync = Lipsync()
        devices = sd.query_devices()

        # self.audio_device = sd.default.device[1]
        # self.music = sf.read("game_jam3.wav", dtype="float32")
        

        # self.audio_device = QAudioOutput()
        # self.music = QMediaPlayer()
        # self.music.setAudioOutput(self.audio_device)
        # self.music.setSource(QUrl.fromLocalFile("game_jam3.wav"))
        # self.music.setLoops(-2)
        # self.music.play()
        # self.voice = QSoundEffect()
        mixer.init()
        mixer.music.load("game_jam3.wav")
        mixer.music.play(loops=-1)
        mixer.music.set_volume(0.25)

        # sd.play(self.music[0]*self.volume, samplerate=44100, loop=True)

        fish_thread = threading.Thread(target=self.try_speak_loop, daemon=True)
        fish_thread.start()

    def set_audio_false(self):
        self.curr_playing = False

    def nightcore(self, sample,sr):
        return pyrb.time_stretch(np.asarray(sample), sr, 2.0)

    def gpt(self, prompt):
        # try:
        #     completion = self.client.chat.completions.create(
        #         model="text-davinci-003",
        #         messages=[
        #             {"role": "system", "content": "You are a helpful assistant."},
        #             {"role": "user", "content": "Hello!"}
        #         ]
        #     )

        #     with open("text.txt", "w") as f:
        #         f.write(completion.choices[0].message)

        #     return completion.choices[0].text
        # except openai.RateLimitError as e:
        #     print(f"Rate limit error: {e}")
        #     print("You may have exceeded your API usage quota. Check your OpenAI dashboard for details.")
        #     return None
        # except Exception as e:
        #     print(f"Error during GPT API request: {e}")
        #     return None

            completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a robot on a screen. The player is throwing contaminated objects at you. You are very angry and hate the player, often using joking language to express your emotions but very angry. NO SWEARING. You also often threaten the player. You *MUST* refer directly to the objects the player throws at you in your response. Keep your responses SHORT (a sentence). NO SWEARING!"},
                {"role": "user", "content": prompt}
            ]
            )

            print(completion.choices[0].message.content)
            # save to text.txt
            with open("text.txt", "w") as f:
                f.write(completion.choices[0].message.content)

            return completion.choices[0].message.content

    def try_speak(self, prompt):
        self.speak_queue.put(prompt)

    def try_speak_loop(self):
        while True:
            # print("AA")
            # print(self.speak_queue.empty())
            if not self.speak_queue.empty():
                print("A")
                prompt = self.speak_queue.get()
                # process_time does not count sleeping
                if time.time() - self.last_speak > self.delay and not self.curr_playing:
                    print("speaking ")
                    self.speak(prompt)
                else:
                    print("refuse to speak")
                    print(time.time() - self.last_speak)
                    print(self.curr_playing)
            time.sleep(0.01)
                
    def speak(self, prompt):    # returns a list of lipsync values
        self.curr_playing = True
        self.tts.tts_to_file(text=prompt, file_path="speech.wav")
        # array = self.tts.tts(text=prompt, speaker=self.tts.speakers[17])
        # tts_speak = sf.read("speech.wav", dtype="float32")
        with open("text.txt", "w") as f:
            f.write(prompt)
        
        # print(self.lipsync.get_features())
        features = self.lipsync.get_features()
        self.lip_queue.put(features)
        # send info to change facial features

        # sd.play(tts_speak[0]*self.volume, 22050)
        # self.voice.setSource(QUrl.fromLocalFile("speech.wav"))
        # print("before play")
        # self.voice.play()
        # while self.voice.isPlaying():
        #     time.sleep(0.1)
        # print("played")

        voice = mixer.Sound("speech.wav")
        voice.play()
        # time.sleep(0)
        self.last_speak = time.time()
        self.curr_playing = False
        # array = self.nightcore(array, 22050)
        # print(self.audio)
        # TextToSpeech.curr_playing = True
        # make audio from gpt text
        # thisll take a while, you can do shit as it renders

        # array = self.tts.tts(text=gpt(prompt), speaker=tts.speakers[17])
        # self.audio = self.tts.tts(text=prompt, speaker=self.tts.speakers[17])
        # interesting sounding vctks: 3,13,14,17,
        # 75, 65, 74, 73, 98, 64

        # nightcore
        # array = nightcore(array,22050)

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    sound = Sound()
    time.sleep(1)
    sound.try_speak("hello there... HELLO")
    print("sent hello!!!!!")
    time.sleep(10)
    print("sent hello 2 !!!!!!")
    sound.try_speak("resignlob")
    sound.try_speak("AAAA")
    time.sleep(5)
    sound.try_speak("testing")
    time.sleep(6)
    sound.try_speak("WHY")
    app.exec()
    # app = QGuiApplication(sys.argv)
    # print("a")
    # e =  QSoundEffect()
    # e.setSource(QUrl.fromLocalFile("speech.wav"))
    # e1 = QSoundEffect()
    # e1.setSource(QUrl.fromLocalFile("game_jam3.wav"))
    # e.play()
    # e1.play()
    # app.exec()

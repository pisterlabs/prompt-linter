#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-03-26 14:36:41 (ywatanabe)"
import os
import warnings

import openai
import speech_recognition as sr

# import whisper
from gtts import gTTS


def s2t_unpaid(lpath_wav, language="EN"):
    """
    Speech to Text using a Google's product.

    Example:
        from scripts import audio, ml
        tmp_wav="/tmp/tmp.wav"
        audio.rec_wav_unlim(spath=tmp_wav)
        said = ml.s2t_unpaid(tmp_wav, language="EN")
        print(said)
        audio.play_audio(tmp_wav)
    """

    r = sr.Recognizer()
    with sr.AudioFile(lpath_wav) as source:
        audio = r.record(source)
    try:
        said_text = r.recognize_google(audio, language=language)
        return said_text
    except Exception as e:
        print(e)


class ChatGPT(object):
    def __init__(
        self,
        system_setting="Please answer in a conversational manner as if you are my friend. "
        "I would be grateful if your response would be less than 20 words.",
    ):
        self.counter = 0
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.OPENAI_API_KEY

        if self.OPENAI_API_KEY is None:
            print('\n"OPENAI_API_KEY" is not set as an environmental varible.\n')

        self.chat_history = []
        self.chat_history.append(
            {
                "role": "system",
                "content": system_setting,
            }
        )

    def __call__(self, text):
        self.counter += 1

        if self.OPENAI_API_KEY is None:
            return "OPENAI_API_KEY is not set as an environmental variable"

        if text is None:
            text = ""
        try:
            self.chat_history.append({"role": "user", "content": text})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.chat_history,
                request_timeout=10,
                # max_tokens=30,
            )
            out_text = response["choices"][0]["message"]["content"]
            self.chat_history.append({"role": "assistant", "content": out_text})

            return out_text
        except Exception as e:
            print(e)
            out_text = "I could not catch you well."
            self.chat_history.append({"role": "assistant", "content": out_text})
            return out_text


def t2s(text, spath="/tmp/t2s.mp3", print_save=False):
    """
    Text to Speech using a Google's product.

    Example:
        from scripts import audio, ml
        # import utils
        text = "This is the test of my t2s function."
        spath="/tmp/t2s.mp3"
        ml.t2s(text, spath=spath)
        audio.play_audio(spath)
    """
    speech = gTTS(text=text, lang="en", slow=False)
    speech.save(spath)
    if print_save:
        print(f"Saved to: {spath}")


# $ pip install git+https://github.com/openai/whisper.git
# class Whisper(object):
#     """
#     Speach to text using Whisper by OpenAI.

#     from scripts import audio, ml

#     spath = "/tmp/test.wav"
#     audio.rec_wav_unlim(spath=spath, fs=44100)

#     mywhisper = Whisper()
#     mywhisper(spath)
#     """

#     def __init__(
#         self,
#     ):
#         self.model = whisper.load_model("base")

#     def __call__(self, lpath):
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", UserWarning)
#             result = self.model.transcribe(lpath)
#         return result["text"]

# Code authors: Masum Hasan, Cengiz Ozel, Sammy Potter
# ROC-HCI Lab, University of Rochester
# Copyright (c) 2023 University of Rochester

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.


import platform
import json
import os
import sys
import time
from pathlib import Path
import openai
from .keys import *
from threading import Lock
from pathlib import Path

operating_system = platform.system()
wav_lock = Lock()

MAX_INSTANCES = 2

root_path = Path(__file__).parent.parent.absolute()
print("root_path: ", root_path)

openai.api_type = os.environ["api_type"]
openai.api_base = os.environ["api_base"]
openai.api_version = os.environ["api_version"]
openai.api_key = os.environ["azure_openai_key"]

# audiodir = os.path.join(root_path, "audio\\instance1\\generated_speech\\")
# audiodir = root_path / Path("audio/instance1/generated_speech/")
# audiofile = audiodir / "bot_speech.wav"
emotion_ready = ["NEUTRAL"]
audio_ready_to_send = [False]

local = True

# local_mode_path = os.path.join(root_path, 'files\\local_mode.json')
local_mode_path = root_path / Path('files/local_mode.json')
## Load debug_mode.json and get the "local" value
with open(local_mode_path) as json_file:
    local_mode = json.load(json_file)
    local = local_mode["local"]

# iframe_url = "https://sapien.coach:81"
iframe_url = "https://sapien.coach:"
if local:
    iframe_url = "http://localhost:"

# Redirect all print to log file
# if not local:
#     with open("log/messages.log", 'w') as sys.stdout:
#         print(time.time())

# Send blendshapes
if local:
    play_audio = False
    send_blendshapes = True
else:
    play_audio = False
    send_blendshapes = True


SSML_mapping = {
    "NEUTRAL": "",
    "HAPPY": "friendly",
    "SAD": "sad",
    "ANGRY": "angry",
    "SURPRISED": "excited",
    "AFRAID": "sad",
    "DISGHUSTED": "sad"
}

Expressions = {
    'NEUTRAL': 0,
    'HAPPY' : 1,
    'SAD' : 2,
    'ANGRY' : 3,
    'SURPRISED' : 4,
    'DISGUSTED' : 5,
    'AFRAID' : 6
}

# languages_path = os.path.join(root_path, 'files\\languages.json')
languages_path = root_path / Path('files/languages.json')
with open(languages_path, 'r', encoding='utf-8') as f:
    languages = json.load(f)

# https://docs.microsoft.com/en-US/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python&tabs=terminal
male_voices = {}
female_voices = {}
all_voices = {}

for lang_code in languages.keys():
   details = languages[lang_code]
   male_voices[lang_code] = details["male"]
   female_voices[lang_code] = details["female"]
   all_voices[lang_code] = details["male"] + details["female"]

vocal_mapping = {
    'him': male_voices,
    'he': male_voices,
    'she': female_voices,
    'her': female_voices,
    'he/him': male_voices,
    'she/her': female_voices,
    'they/them': all_voices,
    'they': all_voices,
    'them': all_voices,
}


emoji_dict = {'😂': 'HAPPY_high',
 '🤣': 'HAPPY_high',
 '🥳': 'HAPPY_high',
 '🤩': 'HAPPY_high',
 '🥰': 'HAPPY_high',
 '😄': 'HAPPY_medium',
 '😁': 'HAPPY_medium',
 '😆': 'HAPPY_medium',
 '😃': 'HAPPY_medium',
 '🤗': 'HAPPY_medium',
 '😍': 'HAPPY_medium',
 '🤠': 'HAPPY_medium',
 '🤓': 'HAPPY_medium',
 '🙂': 'HAPPY_low',
 '😊': 'HAPPY_low',
 '😌': 'HAPPY_low',
 '😉': 'HAPPY_low',
 '👍': 'HAPPY_low',
 '😇': 'HAPPY_low',
 '😅': 'HAPPY_low',
 '🙃': 'HAPPY_low',
 '😘': 'HAPPY_low',
 '😭': 'SAD_high',
 '😿': 'SAD_high',
 '😞': 'SAD_high',
 '😫': 'SAD_high',
 '🤧': 'SAD_high',
 '😢': 'SAD_medium',
 '💔': 'SAD_medium',
 '🥺': 'SAD_medium',
 '😥': 'SAD_medium',
 '😓': 'SAD_medium',
 '😣': 'SAD_medium',
 '😖': 'SAD_medium',
 '😔': 'SAD_low',
 '☹️': 'SAD_low',
 '😕': 'SAD_low',
 '😟': 'SAD_low',
 '🥲': 'SAD_low',
 '🙁': 'SAD_low',
 '😲': 'SURPRISED_high',
 '😵‍💫': 'SURPRISED_high',
 '😯': 'SURPRISED_high',
 '😮': 'SURPRISED_high',
 '🤯': 'SURPRISED_high',
 '😳': 'SURPRISED_medium',
 '😦': 'SURPRISED_medium',
 '😧': 'SURPRISED_medium',
 '🙀': 'SURPRISED_medium',
 '🤭': 'SURPRISED_low',
 '😱': 'AFRAID_high',
 '😨': 'AFRAID_high',
 '👻': 'AFRAID_high',
 '😰': 'AFRAID_medium',
 '😵': 'AFRAID_low',
 '🙈': 'AFRAID_low',
 '😡': 'ANGRY_high',
 '👿': 'ANGRY_high',
 '💢': 'ANGRY_high',
 '🤬': 'ANGRY_high',
 '☠': 'ANGRY_high',
 '😠': 'ANGRY_medium',
 '😾': 'ANGRY_medium',
 '😤': 'ANGRY_medium',
 '🙎': 'ANGRY_medium',
 '🙎‍♂️': 'ANGRY_medium',
 '🙎‍♀️': 'ANGRY_medium',
 '😒': 'ANGRY_low',
 '🙄': 'ANGRY_low',
 '😑': 'ANGRY_low',
 '🤮': 'DISGUSTED_high',
 '🤢': 'DISGUSTED_high',
 '😝': 'DISGUSTED_high',
 '😬': 'DISGUSTED_medium',
 '🥵': 'DISGUSTED_medium'}
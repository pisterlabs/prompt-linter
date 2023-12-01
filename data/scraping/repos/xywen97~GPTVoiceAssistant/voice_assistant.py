

import pyaudio
import wave
import time
import os
import sys
import threading
import queue
import datetime
import ipywidgets as widgets
import websocket
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import numpy as np
import openai
import pyttsx3
import ctypes
import inspect
import pygame

# Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "tmp.pcm"
wsParam = None

STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1 
STATUS_LAST_FRAME = 2
my_saying = ""

THRESHOLD = 500  # The threshold intensity that defines silence signal (lower than).
frames = []
messages = []
silence_flag = False
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,input_device_index=1,
                frames_per_buffer=CHUNK)
speak_thread = None

class Ws_Param(object):
    # Initialize
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile

        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo":1,"vad_eos":10000}

    # Generate url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # Get current time
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # concat string to generate signature origin string
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # use hmac-sha256 encrypt, and encode as base64
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # make a dict
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # put dict into url
        url = url + '?' + urlencode(v)
        return url


# 收到websocket消息的处理
def on_message(ws, message):
    try:
        code = json.loads(message)["code"]
        sid = json.loads(message)["sid"]
        if code != 0:
            errMsg = json.loads(message)["message"]
            # print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        else:
            data = json.loads(message)["data"]["result"]["ws"]
            # print(data)
            # print(json.loads(message))
            result = ""
            for i in data:
                for w in i["cw"]:
                    result += w["w"]
            # print("sid:%s call success!,data is:%s" % (sid, json.dumps(data, ensure_ascii=False)))
            global my_saying
            for i in data:
                for w in i["cw"]:
                    my_saying += w["w"]
            # print(my_saying)
    except Exception as e:
        print("receive msg,but parse exception:", e)

def on_error(ws, error):
    print("### error:", error)

def on_close(ws,a,b):
    # print("### closed ###")
    pass



# websocket handler
def on_open(ws):
    def run(*args):
        global wsParam
        frameSize = 8000  # size of each frame of audio data
        intervel = 0.04  # interval between two frames of audio data
        status = STATUS_FIRST_FRAME  # status of audio data, 0 for first frame, 1 for continue frame, 2 for last frame

        with open(wsParam.AudioFile, "rb") as fp:
            while True:
                buf = fp.read(frameSize)
                # end of file
                if not buf:
                    status = STATUS_LAST_FRAME
                # handle the first frame
                # send data to server, the format of data is json
                # appid is needed here
                if status == STATUS_FIRST_FRAME:

                    d = {"common": wsParam.CommonArgs,
                         "business": wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                # handle the continue frame
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                # handle the last frame
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                # interval between two frames of audio data
                time.sleep(intervel)
        ws.close()

    thread.start_new_thread(run, ())

# For recording, not used
def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,input_device_index=1,
                    frames_per_buffer=CHUNK)
    print("Start recording, please say something ...")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished!")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def recognize():
    global wsParam
    # input your own params
    # you can get these params in xfyun website/科大讯飞开放平台/语音听写
    wsParam = Ws_Param(APPID='xxx', APISecret='xxx',
                       APIKey='xxx',
                       AudioFile=r'tmp.pcm')
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

def gpt(messages):
    openai.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxx'
    global counter
    # global messages
    # messages.append({"role": "user", "content": message})
    # print(message)

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages,
        temperature = 1
    )
    reply = chat.choices[0].message.content

    # messages.append({"role": "assistant", "content": reply})
    # time.sleep(5)
    return reply

pygame.mixer.init()

# 删除temp_*.wav文件
for file in os.listdir():
    if file.startswith("temp_") and file.endswith(".wav"):
        os.remove(file)

counter_file = 0
try:
    while True:
        # time.sleep(0.1)
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        energy = np.sum(np.abs(audio_data)) / CHUNK
        frames.append(data)
        
        # remove old frames every 1000 frames
        if len(frames) > 1000:
            frames = frames[-5:]
        
        if energy > THRESHOLD:
            print("\n ----------------- New chat -----------------\n")
            pygame.mixer.music.stop()
            # start recording
            frames = frames[-5:]
            time_silence = 0
            while True:
                data = stream.read(CHUNK)
                audio_data = np.frombuffer(data, dtype=np.int16)
                energy = np.sum(np.abs(audio_data)) / CHUNK
                
                frames.append(data)
                if energy < THRESHOLD:
                    if not silence_flag:
                        silence_flag = True
                        time_silence = time.time()
                    else:
                        if time.time() - time_silence > 1:
                            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(frames))
                            wf.close()

                            silence_flag = False
                            frames = []

                            break
                        else:
                            pass
                else:
                    silence_flag = False
                    time_silence = 0
            
            # recognize
            recognize()
            if my_saying != "":
                if counter_file >=5:
                    counter_file = 0
                counter_file += 1
                print(f"You: {my_saying}")
                messages.append({"role": "user", "content": my_saying})
                reply = gpt(messages)
                messages.append({"role": "assistant", "content": reply})
                print(f"Assit: {reply}")

                outfile = f"temp_{counter_file}.wav"
                engine = pyttsx3.init()
                engine.save_to_file(reply, outfile)
                engine.runAndWait()
                engine.stop()
                # close the engine
                pygame.mixer.music.load(outfile)
                pygame.mixer.music.play()
                        
                # thread.start_new_thread(run, ())

                my_saying = ""
            else:
                print("Can't recognize, please try again")
            
        else:
            pass

except KeyboardInterrupt:
    # if user hits Ctrl/C then exit and close the stream, pygame
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.mixer.music.stop()

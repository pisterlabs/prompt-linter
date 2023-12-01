import custom_speech_recognition as sr
# import pyaudiowpatch as pyaudio
from datetime import datetime, timedelta
import pyaudio
import numpy as np
import wave
from pydub import AudioSegment
import whisper
import zhconv
import queue
import AudioRecorder
import time
from AudioTranscriber import AudioTranscriber
import threading
import torch
import customtkinter as ctk
import openai




RECORD_TIMEOUT = 3
ENERGY_THRESHOLD = 2000
DYNAMIC_ENERGY_THRESHOLD = False
MODEL = 'small'
openai.api_key = ""
history1 = ''
history2 = ''


def chat(prompt):
    global history1
    global history2

    with open('prompt.txt', 'r') as f:
        pr = str(f.read())
    f.close()

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role": "system", "content": pr}, {"role": "user", "content": history1}, {"role": "assistant", "content": history2}, {"role": "user", "content": prompt}])
    print(f"Answer: \n{completion.choices[0].message.content}")

    with open('code.txt', 'w') as f:
        f.write(completion.choices[0].message.content)
    f.close()

    history2 = completion.choices[0].message.content
    history1 = prompt


class WhisperTranscriber:
    def __init__(self):
        self.audio_model = whisper.load_model(MODEL)

    def get_transcription(self, wav_file_path):
        print('正在识别中')
        result = self.audio_model.transcribe(wav_file_path, fp16=torch.cuda.is_available())
        '''except Exception as e:
            print(e)
            return '''''
        text = zhconv.convert(result['text'].strip(), 'zh-cn')
        if text == '':
            print('内容为空')
        else:
            print(f"You: {text}")
            chat(str(text))





def main():
    print('加载中......')
    model = WhisperTranscriber()
    print("* 模型准备完成...")

    root = ctk.CTk()
    audio_queue = queue.Queue()

    user_audio_recorder = AudioRecorder.DefaultMicRecorder()
    user_audio_recorder.record_into_queue(audio_queue)
    
    global transcriber
    transcriber = AudioTranscriber(user_audio_recorder.source,model)
    global last_spoken
    transcribe = threading.Thread(target=transcriber.transcribe_audio_queue, args=(audio_queue,))
    transcribe.daemon = True
    transcribe.start()


    root.withdraw()
    root.mainloop()


if __name__ == "__main__":
    main()

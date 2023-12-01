import torch
import openai
import whisper
import queue
import speech_recognition as sr
import threading
import numpy as np
import time
from threading import Event, Thread
import asyncio

class Whisper_STT:
    def __init__(self,model="base",device=("cuda" if torch.cuda.is_available() else "cpu"),\
                 energy=300,pause=0.5,save_file=False,model_root="./.cashe/whisper",mic_index=None):
        self.energy = energy
        self.pause = pause
        self.save_file = save_file # 아직 안쓰임
        self.options = {'task': 'transcribe', 'language': 'Korean'}

        self.audio_model = whisper.load_model(model, download_root=model_root).to(device)
        self.audio_queue = queue.Queue()
        self.result_queue: "queue.Queue[str]" = queue.Queue()

        self.break_threads = False
        self.banned_results = [""," ","\n",None]

        self.setup_mic(mic_index)

    def setup_mic(self, mic_index):
        if mic_index is None:
            print("No mic index provided, using default")
        self.source = sr.Microphone(sample_rate=16000, device_index=mic_index)

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy
        self.recorder.pause_threshold = self.pause

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=3)
        print("Mic setup complete, Listening for wakeup word")

    def preprocess(self, data):
        return torch.from_numpy(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0)

    def get_all_audio(self, min_time: float = 2.):
        audio_list = []  # Use a list to collect bytes
        time_start = time.time()

        while time.time() - time_start < min_time:
            got_audio = False  # Initialize the flag within the loop
            while not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_list.append(audio_data)  # Append bytes to the list
                got_audio = True

            if got_audio:
                time_start = time.time()  # Reset the timer if we got audio

        # Clear the queue to prevent old audio data from accumulating
        while not self.audio_queue.empty():
            self.audio_queue.get()

        audio = b''.join(audio_list)  # Join all bytes into a single bytes object

        data = sr.AudioData(audio, 16000, 2)
        data = data.get_raw_data()
        return data


    def record_callback(self,_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)

    def transcribe_forever(self) -> None:
        while True:
            if self.break_threads:
                break
            self.transcribe()

    def transcribe(self,data=None) -> None:
        if data is None:
            audio_data = self.get_all_audio()
        else:
            audio_data = data
        audio_data = self.preprocess(audio_data)

        result = self.audio_model.transcribe(audio_data, **self.options)
        predicted_text = result["text"]
        if predicted_text not in self.banned_results:
            self.result_queue.put_nowait(predicted_text)

    def listen_loop(self) -> None:
        threading.Thread(target=self.transcribe_forever).start()
        while True:
            result = self.result_queue.get()
            print(result)

    def listen(self, timeout: int = 4):
        audio_data = self.get_all_audio(timeout)
        self.transcribe(data=audio_data)
        while True:
            if not self.result_queue.empty():
                return self.result_queue.get()

class Filter_STT(Whisper_STT):
    def __init__(self, model="base", device=("cuda" if torch.cuda.is_available() else "cpu"),\
                 energy=300, pause=0.5, save_file=False, model_root="./.cache/whisper", mic_index=None,\
                 wakeup_word='안녕'):
        super().__init__(model, device, energy, pause, save_file, model_root, mic_index)
        self.wakeup_event = Event() 
        self.thread_running = Event() 
        self.wakeup_word = wakeup_word
        self.destination_list = ['미래관', '정문', '후문', '본관', '학생회관', '시대융합관', '창공관']
        self.WAKEUP_WORD_PROMPT = f"한국어로 대화. {self.wakeup_word}이라는 wakeup-word를 탐지해. \
                                   입력값 중에 {self.wakeup_word}과 같거나 유사한 게 있으면 Yes를 대답. \
                                   이외의 입력값은 No로 대답"
        self.PROMPT_FOR_SYSTEM = f"한국어로 대화. 너는 서울시립대학교 캠퍼스 홍보 및 길 안내 로봇이다.\
                                  If user content가 {self.destination_list}에 있으면 'M:(리스트의 해당 목적지 index 번호)'로 대답. \
                                  Else if user content가 빨리 가라는 내용이면 'SF'라고만 대답.\
                                  Else if user content가 천천히 가라는 내용이면 'SL'라고만 대답.\
                                  Else user content은 'Q:user content' 로 똑같이 대답."
        
    def chat_completion(self, model, temp, messages):
        return openai.ChatCompletion.create(
            model=model,
            temperature=temp,
            messages=messages
    )['choices'][0]['message']['content']

    # STT_Agent의 detect_wakeup_word 메소드
    def detect_wakeup_word(self, predicted_text):
        messages = [
            {"role": "system", "content": self.WAKEUP_WORD_PROMPT},
            {"role": "user", "content": predicted_text},
        ]
        return self.chat_completion("gpt-3.5-turbo", 0, messages) == "Yes"

    # STT_Agent의 preprocess_transcript 메소드
    def preprocess_text(self, predicted_text):
        messages = [
            {"role": "system", "content": self.PROMPT_FOR_SYSTEM},
            {"role": "user", "content": "동아리는 뭐가 있어?"},
            {"role": "assistant", "content": "Q:동아리는 뭐가 있어?"},
            {"role": "user", "content": "미래관으로 가줘"},
            {"role": "assistant", "content": "M:미래관"},
            {"role": "user", "content": "속도 줄여줘"},
            {"role": "assistant", "content": "SL"},
            {"role": "user", "content": predicted_text},
        ]
        return self.chat_completion("gpt-3.5-turbo", 0, messages)
    
    def listening_for_wakeup_word(self):
        self.thread_running.set() # thread_running을 True로 바꿈
        while self.thread_running.is_set(): # thread_running이 True인 동안 반복
            print('---------------------------------')
            audio_data = self.get_all_audio()
            audio_data = self.preprocess(audio_data)
            result = self.audio_model.transcribe(audio_data, initial_prompt=self.wakeup_word)
            predicted_text = result["text"]

            if predicted_text not in self.banned_results:
                if self.detect_wakeup_word(predicted_text):
                    print("Wakeup word detected!: ", predicted_text)
                    self.wakeup_event.set()  # wakeup_event가 True로 바뀜
                    self.thread_running.clear() # thread_running을 False로 바꿈
                    return
            else:
                print("No wakeup word detected. Input :", predicted_text)

    def stop_listening(self):
        self.thread_running.clear()  # Stop the thread

    def listen_for_task(self):
        time.sleep(0.5)
        print("Listening...")
        audio_data = self.get_all_audio()
        audio_data = self.preprocess(audio_data)
        result = self.audio_model.transcribe(audio_data)
        predicted_Q = result["text"]

        print("Predicted_Q = ", predicted_Q)
        if predicted_Q not in self.banned_results:
            processed_Q = self.preprocess_text(predicted_Q)
            return processed_Q  
        else:
            return None


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    stt = Filter_STT()
    
    while True:
        stt.wakeup_event.clear() # wakeup_event를 False로 초기화
        thread = Thread(target=stt.listening_for_wakeup_word)
        thread.start()
        stt.wakeup_event.wait() # wakeup_event가 True가 될 때까지 대기

        stt.stop_listening()
        thread.join()

        processed_Q = stt.listen_for_task()
        print("Processed Message = ", processed_Q)  # 실제 main 함수에서는 processed_Q가 LLM에 전달됨

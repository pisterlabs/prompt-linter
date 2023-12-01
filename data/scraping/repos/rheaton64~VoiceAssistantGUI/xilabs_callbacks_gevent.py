import queue
import gevent
from gevent.event import Event
from elevenlabs import generate, stream, set_api_key, voices
from langchain.callbacks.base import BaseCallbackHandler

audio_queue = queue.Queue()
playing_audio = Event()

# Generates an audio stream from text and adds it to the audio queue
def enqueue_generation(text: str, voice: str):
    audio = generate(
        text=text,
        voice=voice,
        stream=True,
    )
    audio_queue.put(audio)
    if not playing_audio.is_set():
        gevent.spawn(play_audio_from_queue)

# Plays audio from the audio queue
def play_audio_from_queue():
    while True:
        audio = audio_queue.get()
        playing_audio.set()
        stream(audio)
        if audio_queue.qsize() == 0:
            playing_audio.clear()
        audio_queue.task_done()

class XILabsCallbackHandler(BaseCallbackHandler):

    def __init__(self, voice, api_key):
        set_api_key(api_key)
        print('set API key')
        gevent.Greenlet.spawn(voices)
        print('got voices')
        self.output_handler = XILabsOutputHandler(voice)
        print('created output handler')
        

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print('Got token:', token)
        self.output_handler.send_token(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.output_handler.flush()

class XILabsOutputHandler():
    def __init__(self, voice):
        self.voice = voice
        self.token_buffer = []
        self.sentence_buffer = ""
        self.code_buffer = ""
        self.is_code = False
        self.is_code_end = False
        self.is_incomplete_delimiter = False
        self.num_curr_ticks_delimiter = 0
        
    def send_token(self, token):
        if token.startswith("`"): # Check if token starts with a backtick
            if self.num_curr_ticks_delimiter == 0: # Check if it's a new delimiter
                self.is_incomplete_delimiter = True # Set incomplete delimiter flag
                self.num_curr_ticks_delimiter = len(token) # Update current ticks
            else: # If we're in the middle of processing an incomplete delimiter
                self.num_curr_ticks_delimiter += len(token) # Update current ticks
            # Check if we've completed a code block delimiter
            if self.num_curr_ticks_delimiter >=3:
                self.is_incomplete_delimiter = False
                self.num_curr_ticks_delimiter = 0

                if not self.is_code:  # If it's the start of a code snippet
                    self.is_code = True
                    if self.token_buffer:  # If there's text in the buffer
                        self.sentence_buffer += ''.join(self.token_buffer) # Add it to the sentence buffer
                        self.token_buffer.clear()
                    self.sentence_buffer += " I'm writing the code to the window now..." # Add a message to the sentence buffer
                    gevent.Greenlet.spawn(enqueue_generation, self.sentence_buffer, self.voice)
                    self.sentence_buffer = "" # Clear the sentence buffer
                else:  # If it's the end of a code snippet
                    self.is_code_end = True

        if self.is_code:  # If the LLM is currently outputting a code snippet
            self.code_buffer += token

            # Send code buffer if necessary

            if self.is_code_end:  # If it's the end of the code snippet
                self.is_code = False
                self.is_code_end = False
        else: # 
            self.token_buffer.append(token)
            if token.endswith(('.', '?', '!', '"', '\n', ':')):
                self.sentence_buffer += ''.join(self.token_buffer)
                self.token_buffer.clear()
            if self.sentence_buffer and audio_queue.qsize() == 0 and not playing_audio.is_set():
                gevent.Greenlet.spawn(enqueue_generation, self.sentence_buffer, self.voice)
                self.sentence_buffer = ""

    def flush(self):
        if self.sentence_buffer:
            gevent.Greenlet.spawn(enqueue_generation, self.sentence_buffer, self.voice)
            self.sentence_buffer = ""
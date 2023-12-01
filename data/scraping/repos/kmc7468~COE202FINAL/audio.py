import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import io
import math
import openai
import pyaudio
import threading
import time
import wave

SAMPLEING_RATE = 48000
BIT_DPETH = 16
CHUNK_SIZE = 1024

def getbuffersize(duration: float) -> int:
	return math.ceil(SAMPLEING_RATE * BIT_DPETH // 8 * duration)

class AudioCircularBuffer:
	def __init__(self, duration: float):
		self.__index = 0
		self.__size = getbuffersize(duration)
		self.__buffer = bytearray(self.__size)
		self.__lock = threading.Lock()

	def read(self, duration: float = None) -> bytes:
		if duration is None:
			with self.__lock:
				return bytes(self.__buffer[self.__index:]) + bytes(self.__buffer[:self.__index])
		else:
			return self.read()[-getbuffersize(duration):]

	def write(self, data: bytes):
		with self.__lock:
			for value in data:
				self.__buffer[self.__index] = value
				self.__index = (self.__index + 1) % self.__size

class Audio:
	def __init__(self, apikey: str = OPENAI_API_KEY):
		self.__client = openai.OpenAI(
			api_key=apikey
		)
		self.__pyaudio = pyaudio.PyAudio()

	def __del__(self):
		self.stoprecord()

		self.__pyaudio.terminate()

	def startrecord(self, duration: float):
		self.__stream = self.__pyaudio.open(
			format=pyaudio.paInt16,
			channels=1,
			rate=SAMPLEING_RATE,
			frames_per_buffer=CHUNK_SIZE,
			input=True
		)
		self.__buffer = AudioCircularBuffer(duration)

		self.__thread = threading.Thread(target=self.__record, daemon=True)
		self.__threadevent = threading.Event()

		self.__thread.start()

	def stoprecord(self):
		if hasattr(self, "__stream"):
			self.__threadevent.set()
			self.__thread.join()

	def __record(self) -> io.BytesIO:
		while not self.__threadevent.is_set():
			self.__buffer.write(self.__stream.read(CHUNK_SIZE))

		self.__stream.stop_stream()
		self.__stream.close()
		del self.__stream

	def getrecord(self, duration: float = None, sleep: bool = True) -> io.BytesIO:
		stream = io.BytesIO()
		stream.name = "pyaudio.wav"

		with wave.open(stream, "wb") as wf:
			wf.setnchannels(1)
			wf.setsampwidth(self.__pyaudio.get_sample_size(pyaudio.paInt16))
			wf.setframerate(48000)

			if sleep:
				time.sleep(duration)

			wf.writeframes(self.__buffer.read(duration))

		stream.seek(0)

		return stream

	def stt(self, file, prompt: str = None) -> str:
		transcript = self.__client.audio.transcriptions.create(
			model="whisper-1",
			language="ko",
			file=file,
			prompt=prompt
		)

		return transcript.text

	def play(self, file):
		with wave.open(file, "rb") as wf:
			stream = self.__pyaudio.open(
				format=self.__pyaudio.get_format_from_width(wf.getsampwidth()),
				channels=wf.getnchannels(),
				rate=wf.getframerate(),
				output=True
			)

			data = wf.readframes(CHUNK_SIZE)
			while data:
				stream.write(data)
				data = wf.readframes(CHUNK_SIZE)

			stream.stop_stream()
			stream.close()

		file.seek(0)

	def playasync(self, file):
		thread = threading.Thread(target=self.play, args=(file,), daemon=True)
		thread.start()
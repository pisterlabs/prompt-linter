#!/usr/bin/env python3
#
# natbot.py
#
# Set OPENAI_API_KEY to your API key, and then run this from a terminal.
#

from playwright.sync_api import sync_playwright
import time
from sys import argv, exit, platform
import openai
import os
import json
import requests
from threading import Thread
from threading import Timer
from picovoice import Picovoice
from pvrecorder import PvRecorder
import speech_recognition as sr
from PIL import Image, ImageTk
import tkinter as tk
import platform
import re
import threading

from six.moves import queue
from actor import Actor
from dotenv import load_dotenv

load_dotenv()

log = open("logs.txt", "a")

# black_listed_elements = set(["html", "head", "title", "meta", "iframe", "body", "script", "style", "path", "svg", "br", "::marker",])


class PicovoiceThread(Thread):
	def __init__(self, label, access_key):
		super().__init__()

		# Picovoice access key
		self._access_key = access_key

		# tkinter gui
		self._label = label
		self._width = 350
		self._height = 350

		# speech recognition variables
		self._is_ready = False
		self._stop = False
		self._is_stopped = False

		self.r = sr.Recognizer()

	@staticmethod
	def _keyword_path():
		'''
		Method to retrieve Porcupine wake word.
		'''
		if platform.system() == "Darwin":
			return os.path.join(
				os.path.dirname(__file__),
				"picovoice/hey-osler_en_mac_v2_1_0.ppn")
		elif platform.system() == 'Windows':
			return os.path.join(
				os.path.dirname(__file__),
				"picovoice/hey-osler_en_windows_v2_1_0.ppn")
		else:
			raise ValueError("unsupported platform '%s'" % platform.system())

	@staticmethod
	def _context_path():
		'''
		Method to retrieve Rhino context file (speech-to-intent).
		'''
		if platform.system() == "Darwin":
			return os.path.join(
				os.path.dirname(__file__),
				"picovoice/Clinical-Demo_en_mac_v2_1_0.rhn")
		elif platform.system() == 'Windows':
			return os.path.join(
				os.path.dirname(__file__),
				"picovoice/Clinical-Demo_en_mac_v2_1_0.rhn")
		else:
			raise ValueError("unsupported platform '%s'" % platform.system())


	def _wake_word_callback(self):
		stop_event.clear()
		img = Image.open("osler_images/osler_awake.png").resize((self._width, self._height))
		osler_awake = ImageTk.PhotoImage(img)
		self._label.configure(image=osler_awake)
		self._label.image = osler_awake
		recorder.stop()
		command = self.receive_speech_command()
		if command is not None or command != "asd":
			self.osler_thinking()
			if 'begin' in command and 'consultation' in command:
				actor.transcribe_consultation()
				actor.summarise_transcription()
			else:
				actor.perform_command(command)

		recorder.start()
		self.osler_sleeping()

	def osler_thinking(self):
		img = Image.open("osler_images/osler_thinking.png").resize((self._width, self._height))
		osler_thinking = ImageTk.PhotoImage(img)
		self._label.configure(image=osler_thinking)
		self._label.image = osler_thinking

	def osler_sleeping(self):
		img = Image.open("osler_images/osler_sleep.png").resize((self._width, self._height))
		osler_sleeping = ImageTk.PhotoImage(img)
		self._label.configure(image=osler_sleeping)
		self._label.image = osler_sleeping
		
	# def _inference_callback(self, inference):
	#     print("inference: ", inference)
	#     pprint(inspect.getmembers(inference))
	#     img = Image.open(root_path + "demo_screenshots/osler_sleep.png").resize((self._width, self._height))
	#     osler_sleep = ImageTk.PhotoImage(img)
	#     self._label.configure(image=osler_sleep)
	#     self._label.image = osler_sleep

	#     if inference.is_understood:
	#         self.perform_action(inference.intent)

	def _inference_callback(self, inference):
		pass

	def run(self):
		pv = None
		global recorder
		recorder = None

		global actor
		actor = Actor(stop_event)

		try:
			pv = Picovoice(
				access_key=self._access_key,
				keyword_path=self._keyword_path(),
				porcupine_sensitivity=0.75,
				wake_word_callback=self._wake_word_callback,
				context_path=self._context_path(),
				inference_callback=self._inference_callback)

			print(pv.context_info)

			recorder = PvRecorder(device_index=-1, frame_length=pv.frame_length)
			recorder.start()

			self._is_ready = True

			while not self._stop:
				pcm = recorder.read()
				pv.process(pcm)
		finally:
			if recorder is not None:
				recorder.delete()

		# 	if pv is not None:
		# 		pv.delete()

		self._is_stopped = True

	def is_ready(self):
		return self._is_ready

	def stop(self):
		self._stop = True

	def is_stopped(self):
		return self._is_stopped	
		
	def receive_speech_command(self):
		with sr.Microphone() as source:
			print("Listening...")
			audio = self.r.listen(source)

		# # recognize speech using Sphinx
		# try:
		#     print("Sphinx thinks you said " + r.recognize_sphinx(audio))
		# except sr.UnknownValueError:
		#     print("Sphinx could not understand audio")
		# except sr.RequestError as e:
		#     print("Sphinx error; {0}".format(e))


		# recognize speech using Google Speech Recognition
		try:
			# for testing purposes, we're just using the default API key
			# to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
			# instead of `r.recognize_google(audio)`
			print("Command: " + self.r.recognize_google(audio))
			return self.r.recognize_google(audio)
		except sr.UnknownValueError:
			print("Google Speech Recognition could not understand audio")
			return 'asd'
		except sr.RequestError as e:
			print("Could not request results from Google Speech Recognition service; {0}".format(e))
		


def main():
	#set environment variables
	openai.api_key= os.getenv('OPENAI_API_KEY')
	PV_KEY = os.getenv("PV_KEY")

	def stop_thread():
		global stop_event
		stop_event.set()

	global stop_event
	
	stop_event= threading.Event()
	window = tk.Tk()
	window.title("OSLER Demo")
	width, height = 350, 350
	window.geometry(f"{width}x{height}")
	window.attributes("-topmost", True)

	img = Image.open("osler_images/osler_sleep.png").resize((width, height))
	osler_sleep = ImageTk.PhotoImage(img)
	label = tk.Label(window, image=osler_sleep)
	label.pack(fill=tk.BOTH, expand=True)

	stop_button = tk.Button(window, text="Stop", command=stop_thread)
	stop_button.pack(side=tk.BOTTOM, pady=10)

	picovoice_thread = PicovoiceThread(label, PV_KEY)

	def on_close():
		picovoice_thread.stop()
		while not picovoice_thread.is_stopped():
			pass
		window.destroy()

	window.protocol("WM_DELETE_WINDOW", on_close)

	picovoice_thread.start()
	while not picovoice_thread.is_ready():
		pass

	window.mainloop()

if __name__ == "__main__":
    main()
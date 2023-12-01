#!/usr/bin/env python3
from tkinter import *
import intents
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import base64
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import json
from google.cloud import vision
from PIL import ImageGrab, Image, ImageDraw
import pandas as pd
import numpy as np
import time
import pyautogui
import openai
from PIL import Image, ImageTk
import cv2
# from picovoice import Picovoice
# from pvrecorder import PvRecorder
import pyperclip
import json
import re
import sys
from google.cloud import speech
import pyaudio
from six.moves import queue
import sounddevice as sd
from scipy.io.wavfile import write
# import mutagen
# from mutagen.wave import WAVE
import eyed3
import requests
import epic_screens
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import threading
import subprocess
from dotenv import load_dotenv
from gpt import ask_gpt
import gpt_prompts
from mpyg321.MPyg123Player import MPyg123Player # or MPyg321Player if you installed mpg321
from threading import Thread
import threading
from picovoice import Picovoice
from pvrecorder import PvRecorder
import platform
import pvleopard


load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

LEOPARD = pvleopard.create(access_key=os.environ['PICOVOICE_ACCESS_KEY'], model_path='./picovoice/transcribeUserResponse-leopard-v1.2.0-23-06-26--13-50-08.pv')
PLAYER = MPyg123Player()


def update_chat_text(msg):
	txt.insert(END, "\n" + msg)

class MicrophoneStream(object):
	"""Opens a recording stream as a generator yielding the audio chunks."""

	def __init__(self, rate, chunk):
		self._rate = rate
		self._chunk = chunk

		# Create a thread-safe buffer of audio data
		self._buff = queue.Queue()
		self.closed = True

	def __enter__(self):
		self._audio_interface = pyaudio.PyAudio()
		self._audio_stream = self._audio_interface.open(
			format=pyaudio.paInt16,
			# The API currently only supports 1-channel (mono) audio
			# https://goo.gl/z757pE
			channels=1,
			rate=self._rate,
			input=True,
			frames_per_buffer=self._chunk,
			# Run the audio stream asynchronously to fill the buffer object.
			# This is necessary so that the input device's buffer doesn't
			# overflow while the calling thread makes network requests, etc.
			stream_callback=self._fill_buffer,
		)

		self.closed = False

		return self

	def __exit__(self, type, value, traceback):
		self._audio_stream.stop_stream()
		self._audio_stream.close()
		self.closed = True
		# Signal the generator to terminate so that the client's
		# streaming_recognize method will not block the process termination.
		self._buff.put(None)
		self._audio_interface.terminate()

	def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
		"""Continuously collect data from the audio stream, into the buffer."""
		self._buff.put(in_data)
		return None, pyaudio.paContinue

	def generator(self):
		while not self.closed:
			# Use a blocking get() to ensure there's at least one chunk of
			# data, and stop iteration if the chunk is None, indicating the
			# end of the audio stream.
			chunk = self._buff.get()
			if chunk is None:
				return
			data = [chunk]

			# Now consume whatever other data's still buffered.
			while True:
				try:
					chunk = self._buff.get(block=False)
					if chunk is None:
						return
					data.append(chunk)
				except queue.Empty:
					break

			yield b"".join(data)


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

	@staticmethod
	def _keyword_path():
		'''
		Method to retrieve Porcupine wake word.
		'''
		if platform.system() == "Darwin":
			return os.path.join(
				os.path.dirname(__file__),
				"./picovoice/Hey-Osler_en_mac_v2_2_0.ppn")
		elif platform.system() == 'Windows':
			return os.path.join(
				os.path.dirname(__file__),
				"./picovoice/hey-osler_en_windows_v2_1_0.ppn")
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
				"./picovoice/epic_en_mac_v2_2_0.rhn")
		elif platform.system() == 'Windows':
			return os.path.join(
				os.path.dirname(__file__),
				"./picovoice/Clinical-Demo_en_mac_v2_1_0.rhn")
		else:
			raise ValueError("unsupported platform '%s'" % platform.system())

	def match_intent(self, utterance):
		model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

		intents_ls = [
			intents.START_CONSULTATION_NOTE,
			intents.TRANSCRIBE_CONSULTATION,
			# intents.SUMMARISE_CONSULTATION,
			intents.PLACE_ORDERS,
			intents.FILE_DIAGNOSES
			# intents.ANSWER_QUESTIONS,
			# intents.WRITE_LETTER,
			# intents.QUERY_MEDS,
			# intents.QUERY_ORDERS,
    ]

		intent_embeddings = model.encode(intents_ls)
		utterance_embeddings = model.encode(utterance)
		
		cos_scores = cosine_similarity(utterance_embeddings.reshape(1, -1), intent_embeddings)
		cos_scores_torch = torch.from_numpy(cos_scores)
		cos_max = torch.max(cos_scores_torch).item()
		cos_argmax = torch.argmax(cos_scores_torch, dim=1)
		cos = cos_argmax[0].item()

		intent = intents_ls[cos]
		print(f"Intent matched: {intent}")

		return intent, cos_max

	def _wake_word_callback(self):
		img = Image.open("demo_screenshots/osler_awake_smaller.png")
		osler_awake = ImageTk.PhotoImage(img)
		self._label.configure(image=osler_awake)
		self._label.image = osler_awake

		# receive and transcribe the user utterance. Records for 6 seconds currently. 
		recorder.stop()
		self.get_user_utterance()
		actor.user_utterance_text = self.leopard_transcribe()

		# update the chat interface with the user command
		update_chat_text("You -> " + actor.user_utterance_text)

		# match the utterance to an intent
		intent, score = self.match_intent(actor.user_utterance_text)
		print(score)

		# perform the action if match score above a cosine similarity threshold. Currently set at 0.5
		if float(score) > 0.6:
			# update the chat interface with the interpreted intent
			update_chat_text("OSLER -> It looks like you asked me to perform the task: " + intent)

			# extract mrn from utterance
			if intent == intents.START_CONSULTATION_NOTE:
				actor.global_mrn = actor.extract_mrn_from_utterance(actor.user_utterance_text)
				print('mrn: ', actor.global_mrn)

			self.osler_thinking()
			actor.act(intent)

		else:
			# update the chat interface with message reporting unsupported command
			update_chat_text("OSLER -> This request is not currently supported.")

			#play the audio file
			PLAYER.play_song("no_matched_intent.wav")
			time.sleep(5)

		# resume the recorder
		recorder.start()
		self.osler_sleeping()

	def osler_thinking(self):
		img = Image.open("demo_screenshots/osler_thinking_smaller.png")
		osler_thinking = ImageTk.PhotoImage(img)
		self._label.configure(image=osler_thinking)
		self._label.image = osler_thinking


	def osler_sleeping(self):
		img = Image.open("demo_screenshots/osler_sleep_smaller.png")
		osler_sleeping = ImageTk.PhotoImage(img)
		self._label.configure(image=osler_sleeping)
		self._label.image = osler_sleeping

	def perform_action(self, intent):
		self.osler_thinking()
		recorder.stop()
		actor.act(intent)
		recorder.start()
		self.osler_sleeping()

	def get_user_utterance(self):
		fs = 44100  # Sample rate
		seconds = 6  # Duration of recording

		myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
		sd.wait()  # Wait until recording is finished
		write('user_utterance.wav', fs, myrecording)  # Save as WAV file

	def leopard_transcribe(self):
		transcript, words = LEOPARD.process_file('user_utterance.wav')
		print(transcript)
		for word in words:
			print(
			"{word=\"%s\" start_sec=%.2f end_sec=%.2f confidence=%.2f}"
			% (word.word, word.start_sec, word.end_sec, word.confidence))
		return transcript
		
	def _inference_callback(self, inference):
		pass

	def run(self):
		pv = None
		global recorder
		recorder = None

		global actor
		actor = Actor()

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

			if pv is not None:
				pv.delete()

		self._is_stopped = True

	def is_ready(self):
		return self._is_ready

	def stop(self):
		self._stop = True

	def is_stopped(self):
		return self._is_stopped



class Actor:
	def __init__(self) -> None:
		self.consultation_transcript = ""
		self.transcript_summary = ""
		self.consultation_entities = {'orders': [{'name': 'X-ray of upper chest', 'reason': 'Patient has been experiencing chest pain'}, {'name': 'MRI of stomach', 'reason': "Investigation related to patient's chest pain"}, {'name': '24-hour RVCG', 'reason': 'Further investigation of reported chest pain'}], 'medicine': [{'name': 'Blood thinners', 'dosage': 'Unspecified', 'reason': 'Chest pain'}, {'name': 'Ibuprofen', 'dosage': 'Unspecified', 'reason': 'Chest pain'}], 'visit_diagnoses': []}
		# self.consultation_entities = {}
		self.mrn_flag = False
		self.user_utterance_text = ''
		self.patient_mrn_str = ''
		self.patient_mrn_digits = '111'
		self.med_hx = ''
		self.letters_hx = ''
		self.global_mrn = ''
		self.consultation_done = False

	def get_element_center(self, loc):
		'''
		Method to get the center of element's bounding box
		'''
		corner_x, corner_y = loc[0], loc[1]
		width, height = loc[2], loc[3]
		x, y = corner_x/2 + width/4, corner_y/2 + height/4
		return x, y

	def click_screenshot(self, screenshot, confidence=0.8):
		'''
		Method to click on a matching screenshot.
		'''
		# loc = pyautogui.locateOnScreen(root_path + f"demo_screenshots/{screenshot}", confidence=confidence)
		loc = pyautogui.locateOnScreen(f"demo_screenshots/{screenshot}")
		if loc is None:
			print('cant find it!')
			return 0
			# raise Exception("Matching image not found on screen.")
		x, y = self.get_element_center(loc)
		print(f"Mouse click at: {x, y}")
		pyautogui.click(x, y)
		return 1
		
	def activate_application(self, app_name):
		applescript_code = f'''
		tell application "{app_name}"
			activate
		end tell
		'''

		process = subprocess.Popen(['osascript', '-e', applescript_code],
								stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		output, error = process.communicate()

		if error:
			print("Error executing AppleScript:", error)
			return

	def new_tab(self):
		'''
		Opens a new tab.
		'''
		pyautogui.hotkey("ctrl", "t")

	def type_string(self, char_string, interval=0.2):
		'''
		Types a given string.
		'''
		pyautogui.write(char_string, interval=interval)

	def press_key(self, key, presses=1):
		'''
		Presses a given key.
		'''
		pyautogui.press(key, presses=presses)

	def press_command(self, command):
		'''
		Performs a given hotkey command.
		'''
		if command == "copy":
			pyautogui.hotkey("ctrl", "c")
		elif command == "paste":
			pyautogui.hotkey("ctrl", "v")
		elif command == "tab_back":
			pyautogui.hotkey("alt", "tab")
		else:
			raise Exception(f"Command {command} not recognized.")

	def scroll(self, offset):
		'''
		Vertical scrolling.
		'''
		pyautogui.scroll(offset)

	def _wake_word_callback(self):
		img = Image.open("demo_screenshots/osler_awake_smaller.png")
		osler_awake = ImageTk.PhotoImage(img)
		self._label.configure(image=osler_awake)
		self._label.image = osler_awake

	def listen_print_loop(self, responses):
		"""Iterates through server responses and prints them.

		The responses passed is a generator that will block until a response
		is provided by the server.

		Each response may contain multiple results, and each result may contain
		multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
		print only the transcription for the top alternative of the top result.

		In this case, responses are provided for interim results as well. If the
		response is an interim one, print a line feed at the end of it, to allow
		the next result to overwrite it, until the response is a final one. For the
		final one, print a newline to preserve the finalized transcription.
		"""
		num_chars_printed = 0
		for response in responses:
			if not response.results:
				continue

			# The `results` list is consecutive. For streaming, we only care about
			# the first result being considered, since once it's `is_final`, it
			# moves on to considering the next utterance.
			result = response.results[0]
			if not result.alternatives:
				continue

			# Display the transcription of the top alternative.
			transcript = result.alternatives[0].transcript

			# Display interim results, but with a carriage return at the end of the
			# line, so subsequent lines will overwrite them.
			#
			# If the previous result was longer than this one, we need to print
			# some extra spaces to overwrite the previous result
			overwrite_chars = " " * (num_chars_printed - len(transcript))

			if not result.is_final:
				# sys.stdout.write(transcript + overwrite_chars + "\r")
				# sys.stdout.flush()

				# num_chars_printed = len(transcript)
				pass

			else:
				
				# print(transcript + overwrite_chars)
				output = transcript + overwrite_chars
				self.consultation_transcript += output
				output = output.lower()

				if "stop recording" in output:
					break

				pyperclip.copy(transcript + overwrite_chars)
				pyautogui.keyDown('command')
				pyautogui.press('v')
				pyautogui.keyUp('command')

				# Exit recognition if any of the transcribed phrases could be
				# one of our keywords.
				if re.search(r"\b(exit|quit)\b", transcript, re.I):
					print("Exiting..")
					break

				num_chars_printed = 0


	def transcribe(self):
		# See http://g.co/cloud/speech/docs/languagesv
		# for a list of supported languages.
		language_code = "en-US"  # a BCP-47 language tag

		client = speech.SpeechClient()
		config = speech.RecognitionConfig(
			encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
			sample_rate_hertz=RATE,
			language_code=language_code,
			model='medical_conversation'
		)

		streaming_config = speech.StreamingRecognitionConfig(
			config=config, interim_results=True
		)

		with MicrophoneStream(RATE, CHUNK) as stream:
			audio_generator = stream.generator()
			requests = (
				speech.StreamingRecognizeRequest(audio_content=content)
				for content in audio_generator
			)

			responses = client.streaming_recognize(streaming_config, requests)

			# Now, put the transcription responses to use.
			self.listen_print_loop(responses)

	def match_screen(self):
		# get text representation of current screen
		current_screen = ""

		model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

		screens_ls = [
			epic_screens.PATIENT_LOOKUP,
			epic_screens.SCHEDULE,
			epic_screens.PATIENT_PAGE
		]

		epic_embeddings = model.encode(screens_ls)
		screen_embeddings = model.encode(current_screen)

		cos_scores = cosine_similarity(screen_embeddings.reshape(1, -1), epic_embeddings)
		cos_scores_torch = torch.from_numpy(cos_scores)
		cos_max = torch.max(cos_scores_torch).item()
		cos_argmax = torch.argmax(cos_scores_torch, dim=1)
		cos = cos_argmax[0].item()


		print(cos_scores)
		intent = screens_ls[cos]
		print(f"Intent matched: {intent}")

	def act(self, intent):
		if intent == intents.START_CONSULTATION_NOTE:
			self.new_consultation_mrn()
		elif intent == intents.TRANSCRIBE_CONSULTATION:
			self.transcribe_consultation()
			self.consultation_done = True
		elif intent == intents.WRITE_LETTER:
			self.write_referral()
		elif intent == intents.PLACE_ORDERS:
			self.place_orders()
		elif intent == intents.FILE_DIAGNOSES:
			self.file_diagnoses()
		elif intent == intents.LIST_ABILITIES:
			self.list_abilities()
		# elif intent == intents.ANSWER_QUESTIONS:
		#     self.ask_general_consultation_question()
		# elif intent == intents.QUERY_ORDERS:
		#     self.query_orders()
		# elif intent == intents.QUERY_MEDS:
		#     self.query_meds()
		else:
			raise ValueError("unsupported intent '%s'" % intent)
		
		# update to sleeping mode after task done
		picovoice_thread.osler_sleeping()
		
	def get_user_voice_response(self):
		fs = 44100  # Sample rate
		seconds = 6  # Duration of recording

		myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
		sd.wait()  # Wait until recording is finished
		write('user_response.wav', fs, myrecording)  # Save as WAV file

	def leopard_transcribe(self):
		transcript, words = LEOPARD.process_file('user_response.wav')
		print(transcript)
		for word in words:
			print(
			"{word=\"%s\" start_sec=%.2f end_sec=%.2f confidence=%.2f}"
			% (word.word, word.start_sec, word.end_sec, word.confidence))
		return transcript
	
	def str_to_digit(self, nstr):
		digit = ''
		is_digit = True
		if nstr == 'zero':
			digit = '0'
		elif nstr == 'one':
			digit = '1'
		elif nstr == 'two':
			digit = '2'
		elif nstr == 'three':
			digit = '3'
		elif nstr == 'four':
			digit = '4'
		elif nstr == 'five':
			digit = '5'
		elif nstr == 'six':
			digit = '6'
		elif nstr == 'seven':
			digit = '7'
		elif nstr == 'eight':
			digit = '8'
		elif nstr == 'nine':
			digit = '9'
		else:
			print('error converting string to digit')
			is_digit = False
		return digit, is_digit
	
	def convert_string_to_num(self, num_str):
		num_str_ls = num_str.split(' ')
		digits_str = ''
		for num_str in num_str_ls:
			digits_str += self.str_to_digit(num_str)
		return digits_str
	
	def extract_mrn_from_utterance(self, utterance_str):
		str_ls = utterance_str.split(' ')
		mrn = ''
		for s in str_ls:
			digit, is_digit = self.str_to_digit(s)
			if is_digit:
				mrn += digit
		return mrn
	
	def extract_mrn_from_text(self, utterance_str):
		str_ls = utterance_str.split(' ')
		mrn = ''
		for s in str_ls:
			if s.isdigit():
				mrn = s
		return mrn
		
	def ask_general_consultation_question(self):
		# play the audio file of the question
		PLAYER.play_song("ask_general_consultation_question.wav")
		time.sleep(2)

		# record the user response and write to a  wav audio file
		self.get_user_voice_response()

		# use picovoice leopard to transcribe the audio response file
		question = self.leopard_transcribe()

		# combine the quetsion with the consultation transcript
		question_about_consultation_prompt = 'INSTRUCTION: You are a medical doctor who has just performed a consultation and is provided with a transcript of the consultation. Answer a question about the consultation as accurately as possible. The consultation transcritp and question about it will follow\n'
		question_about_consultation_prompt += '\nCONSULTATION TRANSCRIPT: \n' + self.consultation_transcript
		question_about_consultation_prompt += '\nQUESTION ABOUT CONSULTATION: \n' + question + '?\n\n'
		response=openai.Completion.create(
		model="text-davinci-003",
		prompt=question_about_consultation_prompt,
		max_tokens=2500,
		temperature=0
		)
		answer = json.loads(str(response))
		answer = answer['choices'][0]['text']

		# print the answer
		print(answer)

		# create the audio file from the text
		# Language in which you want to convert
		language = 'en'
		
		# Passing the text and language to the engine, 
		# here we have marked slow=False. Which tells 
		# the module that the converted audio should 
		# have a high speed
		myobj = gTTS(text=answer, lang=language, slow=False)
		
		# Saving the converted audio in a mp3 file named
		myobj.save("consulation_answer.wav")

		# get the length of the audio file to know how long to sleep while playing
		audio = eyed3.load("consulation_answer.wav")
		length_in_seconds = int(audio.info.time_secs)

		#play the audio file
		PLAYER.play_song("consulation_answer.wav")
		time.sleep(length_in_seconds + 1)

	def extract_letters(self):
		time.sleep(2)

		pyautogui.keyDown('ctrl')
		pyautogui.press('space')
		pyautogui.keyUp('ctrl')
		time.sleep(1)

		pyperclip.copy('chart review')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')
		time.sleep(1)

		pyautogui.press('down')
		time.sleep(0.5)
		pyautogui.press('enter')
		time.sleep(2)

		self.click_screenshot("letters.png")
		time.sleep(2)
		self.click_screenshot("recent_letters.png")
		time.sleep(1)

		letters = ''

		for i in range(5):
			pyautogui.press("enter")
			time.sleep(2)

			pyautogui.click()
			time.sleep(1)

			pyautogui.keyDown('command')
			pyautogui.press('a')
			pyautogui.keyUp('command')
			time.sleep(1)

			pyautogui.keyDown('command')
			pyautogui.press('c')
			pyautogui.keyUp('command')
			letters += pyperclip.paste()
			time.sleep(1)

			pyautogui.keyDown('option')
			pyautogui.keyDown('command')
			pyautogui.press('left')
			pyautogui.keyUp('option')
			pyautogui.keyUp('command')
			time.sleep(2)

			pyautogui.press('down')
			time.sleep(1)

		self.letters_hx = letters

	def glance_patient_search_results(self):
		# telling the user that a glance is being done
		txt.insert(END, "\n" + "OSLER -> Analysing the screen...")

		parsed_screen = parse_screen()
		sys_instr = gpt_prompts.patient_lookup_outcome

		url = "https://api.openai.com/v1/chat/completions"
		headers = {
			"Content-Type": "application/json",
			"Authorization": "Bearer " + openai.api_key
		}

		conversation = [{"role": "system", "content": sys_instr}]
		conversation.append({"role": "user", "content": parsed_screen})

		payload = {
		"model": "gpt-4-32k",
		"messages": conversation,
		"temperature": 0,
		"max_tokens": 1
		# "stop": "\n"
		}
		response = requests.post(url, headers=headers, json=payload)
		if response.status_code == 200:
			suggested_command = response.json()["choices"][0]["message"]["content"]
			usage = response.json()["usage"]
			return suggested_command, usage
		else:
			print(f"Error: {response.status_code} - {response.text}")

	def new_consultation_mrn(self):
		while True:
			# screenshot and parse current screen
			parsed_screen = parse_screen()
			current_screen = match_screen(parsed_screen)
			txt.insert(END, "\n" + "OSLER -> The current epic screen is: " + current_screen)
			self.activate_application('Citrix Viewer')
			if current_screen == 'schedule':
				# press f10 for search activities bar
				pyautogui.press('f10')
				time.sleep(2)

				# search for write note activity
				pyperclip.copy('write')
				pyautogui.keyDown('command')
				pyautogui.press('v')
				pyautogui.keyUp('command')
				time.sleep(2)

				# press enter to select write note activity
				pyautogui.press('enter')
				time.sleep(2)
			if current_screen == 'patient_lookup':
				print('global_mrn: ', self.global_mrn)
				pyperclip.copy(self.global_mrn)
				pyautogui.keyDown('command')
				pyautogui.press('v')
				pyautogui.keyUp('command')
				time.sleep(2)

				pyautogui.press('enter')
				time.sleep(1)

				# at this point there are three different possible outcomes so need to use UIED to check and handle
				mrn_search_outcome, usage = self.glance_patient_search_results()
				print('mrn search outcome: ', mrn_search_outcome)
				if mrn_search_outcome == '1':
					txt.insert(END, "\n" + "OSLER -> Great! This MRN matches exactly one patient")
					pyautogui.press('enter')
					time.sleep(2)
					pyautogui.press('enter')
					time.sleep(8)
				elif mrn_search_outcome == '2':
					txt.insert(END, "\n" + "OSLER -> Sorry, this MRN matches more than one patient.")
					break
				elif mrn_search_outcome == '3':
					txt.insert(END, "\n" + "OSLER -> Sorry, this MRN does not match any patient. Please try again.")
					break
				else:
					print('error with processing the result from glancing')

			if current_screen == 'chart_review':
				# ctrl space
				pyautogui.keyDown('ctrl')
				pyautogui.press('space')
				pyautogui.keyUp('ctrl')
				time.sleep(2)

				# search for write note activity
				pyperclip.copy('write note')
				pyautogui.keyDown('command')
				pyautogui.press('v')
				pyautogui.keyUp('command')
				time.sleep(2)

				# select write note activity
				pyautogui.press('down')
				time.sleep(1)
				pyautogui.press('enter')
				time.sleep(2)
				pyautogui.press('enter')
				time.sleep(5)

			if current_screen == 'documentation':
				# use the accept button as a unique marker to check if note is already opened
				if not pyautogui.locateOnScreen("demo_screenshots/accept.png", confidence=0.7, grayscale=True):
					self.click_screenshot('create_note.png', confidence=0.6)
					time.sleep(2)

				time.sleep(2)
				pyautogui.press('f3')
				time.sleep(2)

				# release the function button
				pyautogui.keyUp('fn')
				time.sleep(1)

				# add smart text medicines and problem list
				pyautogui.write('.med', interval=0.1)
				time.sleep(1)
				pyautogui.press('enter')
				time.sleep(1)

				# add smart text medicines and problem list
				pyautogui.write('.diagprobap', interval=0.1)
				time.sleep(1)
				pyautogui.press('enter')
				time.sleep(1)

				# copying the patient medical history and medications and saving to memory
				pyautogui.keyDown('command')
				pyautogui.press('a')
				pyautogui.keyUp('command')

				time.sleep(1)

				pyautogui.keyDown('command')
				pyautogui.press('c')
				pyautogui.keyUp('command')

				time.sleep(0.5)
				pyautogui.press('right')

				self.med_hx = pyperclip.paste()
				break


	def transcribe_consultation(self):
		# activate Epic window
		self.activate_application('Citrix Viewer')

		# add header
		pyperclip.copy('\n\n--------- Consultation Transcription ---------\n\n')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

		self.transcribe()

		# stop recording banner
		pyperclip.copy('\n\n--------- Recording Stopped ---------\n\n')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

		self.summarise_transcription()

		self.consultation_entities = ask_gpt(self.consultation_transcript)
		print('extracted metadata from consultation')

		print(str(self.consultation_entities))

	def place_orders(self):
		# check if consultation has been done
		if not self.consultation_done:
			update_chat_text("OSLER -> You have not yet performed a consultation so this request is invalid")
			return

		# bring Epic window to the front
		self.activate_application('Citrix Viewer')

		orders_list = self.get_orders_from_gpt_call(self.consultation_entities)
		for order in orders_list:
			pyautogui.keyDown('command')
			pyautogui.press('o')
			pyautogui.keyUp('command')
			time.sleep(1)

			pyperclip.copy(order)
			pyautogui.keyDown('command')
			pyautogui.press('v')
			pyautogui.keyUp('command')
			time.sleep(0.5)

			pyautogui.press('enter')
			time.sleep(1)
			pyautogui.press('enter')

			pyautogui.keyDown('option')
			pyautogui.keyDown('command')
			pyautogui.press('v')
			pyautogui.keyUp('command')
			pyautogui.keyUp('option')
			time.sleep(1)

		pyautogui.press('escape')

	def list_abilities(self):
		abilities_msg = '''OSLER -> Hi! I'm Osler, your personal AI digital healthcare assistant. I can help you with the following:
		\n- Starting a new consultation note
		\n- Transcribing a consultation
		\n- Placing orders mentioned in the consultation
		\n- Filing diagnoses mentioned in the consultation
		\n- Answering general questions about the consultation
		'''
		update_chat_text(abilities_msg)

	def file_diagnoses(self):
		# check if consultation has been done
		if not self.consultation_done:
			update_chat_text("OSLER -> You have not yet performed a consultation so this request is invalid")
			return

		diagnosis_list = self.get_diagnoses_from_gpt_call(self.consultation_entities)

		# bring Epic window to the front
		self.activate_application('Citrix Viewer')

		pyautogui.keyDown('command')
		pyautogui.press('g')
		pyautogui.keyUp('command')
		time.sleep(1)
		
		for diagnosis in diagnosis_list:
			pyperclip.copy(diagnosis)
			pyautogui.keyDown('command')
			pyautogui.press('v')
			pyautogui.keyUp('command')
			time.sleep(1)

			pyautogui.press('enter')
			time.sleep(1)
			pyautogui.press('enter')
			time.sleep(1)

		pyautogui.press('escape')

	def summarise_transcription(self):
		url = "https://api.openai.com/v1/chat/completions"
		headers = {
			"Content-Type": "application/json",
			"Authorization": "Bearer " + openai.api_key
		}
		SOAP_user_msg_template = """
		MEDICAL HISTORY:
		------------------
		$medical_history
		------------------

		CONSULTATION TRANSCRIPT:
		------------------
		$consultation_transcript
		------------------
		"""

		system_instruction = '''
		You are a medical office assistant drafting documentation for a physician. You will be provided with a MEDICAL HISTORY and a CONSULTATION TRANSCRIPT. DO NOT ADD any content that isn't specifically mentioned in the CONSULTATION TRANSCRIPT or the MEDICAL HISTORY. From the attached transcript and medical history, generate a SOAP note based on the below template format for the physician to review, include all the relevant information and do not include any information that isn't explicitly mentioned in the transcript.If nothing is mentioned just returned[NOT MENTIONED].

		Template for Clinical SOAP Note Format:

		Subjective: The “history” section
		- HPI: include any mentioned symptom dimensions, chronological narrative of patients complains, information obtained from other sources(always identify source if not the patient).
		- Pertinent past medical history.
		- Pertinent review of systems mentioned, for example, “Patient has not had any stiffness or loss
		of motion of other joints.”
		- Current medications mentioned(list with daily dosages).
		Objective: The physical exam and laboratory data section
		- Vital signs including oxygen saturation when indicated.
		- Focussed physical exam.
		- All pertinent labs, x - rays, etc.completed at the visit.
		Assessment / Problem List: Your assessment of the patients problems
		- Assessment: A one sentence description of the patient and major problem
		- Problem list: A numerical list of problems identified
		- All listed problems need to be supported by findings in subjective and objective areas above.Try to take the assessment of the major problem to the highest level of diagnosis that you can, for example, “low back sprain caused by radiculitis involving left 5th LS nerve root.”
		- Any differential diagnoses mentioned in the transcript, if not just leave this blank as DIFFERENTIAL DIAGNOSIS:
		Plan: Any plan for the patient mentioned in the transcript
		- Divide any diagnostic and treatment plans for each differential diagnosis.
		- Your treatment plan should include: patient education pharmacotherapy if any, other therapeutic procedures.You must also address plans for follow - up(next scheduled visit, etc.)
		Please provide your response in a bullet point list for each heading.'''

		user_message = SOAP_user_msg_template
		user_message = user_message.replace("$medical_history", self.med_hx)
		user_message = user_message.replace("$consultation_transcript", self.consultation_transcript)

		conversation = [{"role": "system", "content": system_instruction}]
		conversation.append({"role": "user", "content": user_message})

		payload = {
		"model": "gpt-4",
		"messages": conversation,
		"temperature": 0,
		"max_tokens": 500
		# "stop": "\n"
		}
		response = requests.post(url, headers=headers, json=payload)
		if response.status_code == 200:
			suggested_command = response.json()["choices"][0]["message"]["content"]
			usage = response.json()["usage"]
			# return suggested_command, usage
		else:
			print(f"Error: {response.status_code} - {response.text}")

		
		# write consultation summary to notes
		pyperclip.copy(suggested_command)
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

	
	def summarise_transcription1(self):
		# add header
		pyperclip.copy('\n\n--------- Consultation Summary ---------\n\n')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

		# get GPT consultation summary
		meta_consultation_summarisation = 'INSTRUCTION: Summarise the below MEDICAL HISTORY and CONSULTATION TRANSCRIPT between patient and doctor into short notes, under the following headings: 1. Detailed summary of the patient symptoms  2. Medicines 3. Allergies 4. Family History 5. Social History 6. Examination findings 7. Impression 8. Plan\n'
		meta_consultation_summarisation += 'MEDICAL HISTORY: \n' + self.med_hx
		meta_consultation_summarisation += '\nCONSULTATION TRANSCRIPT: \n' + self.consultation_transcript + '\n\n'
		response=openai.Completion.create(
		model="text-davinci-003",
		prompt=meta_consultation_summarisation,
		max_tokens=2500,
		temperature=0
		)
		consultation_summary = json.loads(str(response))
		consultation_summary = consultation_summary['choices'][0]['text']
		self.transcript_summary = consultation_summary

		# write consultation summary to notes
		pyperclip.copy(consultation_summary)
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

	def get_orders_from_gpt_call(self, response):
		orders_ls = []
		for order in response['orders']:
			orders_ls.append(order['name'])
		return orders_ls
	
	def get_diagnoses_from_gpt_call(self, response):
		diagnoses_ls = []
		for diagnosis in response['visit_diagnoses']:
			diagnoses_ls.append(diagnosis['name'])
		return diagnoses_ls
	
	def get_meds_from_gpt_call(self, response):
		meds_ls = []
		for med in response['medicine']:
			meds_ls.append(med['name'])
		return meds_ls

	def speak_orders_list(self, orders_list):
		# The text to be converted into audio
		text = 'The orders I got from the consultation were '
		for i in range(len(orders_list)):
			text += orders_list[i]
			if i < len(orders_list) - 1:
				text += ' and '
		return text

	def speak_meds_list(self, meds_list):
		# The text to be converted into audio
		text = 'The medicines I got from the consultation were '
		for i in range(len(meds_list)):
			text += meds_list[i]
			if i < len(meds_list) - 1:
				text += ' and '
		return text

	def query_orders(self):
		# get the list of orders extracted from the consultation
		orders_list = self.get_orders_from_gpt_call(self.consultation_entities)

		# convert the list of orders into the text to speak
		audio_text = self.speak_orders_list(orders_list)

		# create the audio file from the text
		# Language in which you want to convert
		language = 'en'
		
		# Passing the text and language to the engine, 
		# here we have marked slow=False. Which tells 
		# the module that the converted audio should 
		# have a high speed
		myobj = gTTS(text=audio_text, lang=language, slow=False)
		
		# Saving the converted audio in a mp3 file named
		myobj.save("extracted_orders_list.wav")

		# get the length of the audio file to know how long to sleep while playing
		audio = eyed3.load("extracted_orders_list.wav")
		length_in_seconds = int(audio.info.time_secs)

		#play the audio file
		PLAYER.play_song("extracted_orders_list.wav")
		time.sleep(length_in_seconds + 1)

	def query_meds(self):
		# get the list of orders extracted from the consultation
		meds_list = self.get_meds_from_gpt_call(self.consultation_entities)

		# convert the list of orders into the text to speak
		audio_text = self.speak_meds_list(meds_list)

		# create the audio file from the text
		# Language in which you want to convert
		language = 'en'
		
		# Passing the text and language to the engine, 
		# here we have marked slow=False. Which tells 
		# the module that the converted audio should 
		# have a high speed
		myobj = gTTS(text=audio_text, lang=language, slow=False)
		
		# Saving the converted audio in a mp3 file named
		myobj.save("extracted_meds_list.wav")

		# get the length of the audio file to know how long to sleep while playing
		audio = eyed3.load("extracted_meds_list.wav")
		length_in_seconds = int(audio.info.time_secs)

		#play the audio file
		PLAYER.play_song("extracted_meds_list.wav")
		time.sleep(length_in_seconds + 1)

	def write_referral(self):
		self.activate_application('Citrix Viewer')

		# press f10 for search activities bar
		pyautogui.press('f10')
		time.sleep(2)

		# search for write note activity
		pyperclip.copy('letter')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')
		pyautogui.press('enter')
		time.sleep(3)

		# input MRN 111
		pyperclip.copy(self.global_mrn)
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')
		time.sleep(3)

		# press enter 3 times
		pyautogui.press('enter')
		time.sleep(3)
		pyautogui.press('enter')
		time.sleep(3)
		pyautogui.press('enter')
		time.sleep(8)

		# select clinic letter
		self.click_screenshot("select_clinic_letter.png", confidence=0.6)
		time.sleep(3)

		# add recipient as patient 1
		pyautogui.keyDown('command')
		pyautogui.keyDown('option')
		pyautogui.press('1')
		pyautogui.keyUp('command')
		pyautogui.keyUp('option')
		time.sleep(3)

		#play the letter pending audio file
		PLAYER.play_song("letter_pending.wav")
		time.sleep(4)

		# get GPT to write referral letter
		referral_letter_prompt = 'Write a letter to the patients GP including all of the following information, include the patients background medical history, medications, a summary of the consultation and a plan:\n\n'
		referral_letter_prompt += self.transcript_summary

		response=openai.Completion.create(
		model="text-davinci-003",
		prompt=referral_letter_prompt,
		max_tokens=1500,
		temperature=0
		)

		referral_letter = json.loads(str(response))
		print(referral_letter['choices'][0]['text'])

		pyautogui.press('tab', presses=10, interval=0.2)
		time.sleep(1)

		pyperclip.copy(referral_letter['choices'][0]['text'])
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')


def init_ml_client(subscription_id, resource_group, workspace):
	return MLClient(
		DefaultAzureCredential(), subscription_id, resource_group, workspace
	)

ml_client = init_ml_client(
	"af5d9edb-37c3-40a4-a58f-5b97efbbac8d",
	"hello-rg",
	"osler-perception"
)

def read_image(path_to_image):
	with open(path_to_image, "rb") as f:
		return f.read()

def predict_image_object_detection_sample(
		ml_client,
		endpoint_name,
		deployment_name,
		path_to_image
):
	request_json = {
		"image" : base64.encodebytes(read_image(path_to_image)).decode("utf-8")
	}	

	request_fn = "request.json"

	with open(request_fn, "w") as request_f:
		json.dump(request_json, request_f)

	response = ml_client.online_endpoints.invoke(
		endpoint_name=endpoint_name,
		deployment_name=deployment_name,
		request_file=request_fn
	)

	detections = json.loads(response)

	return detections

def detect_text(path):
	"""Detects text in the file."""
	client = vision.ImageAnnotatorClient(credentials=credentials)

	with open(path, 'rb') as image_file:
		content = image_file.read()

	image = vision.Image(content=content)

	response = client.text_detection(image=image)
	texts = response.text_annotations
	# print('Texts:')

	# for text in texts:
	#     # print(f'\n"{text.description}"')

	#     vertices = ([f'({vertex.x},{vertex.y})'
	#                 for vertex in text.bounding_poly.vertices])

	#     # print('bounds: {}'.format(','.join(vertices)))

	if response.error.message:
		raise Exception(
			'{}\nFor more info on error messages, check: '
			'https://cloud.google.com/apis/design/errors'.format(
				response.error.message))
		
	return response

# not including bboxes just yet
def html_from_UIE(df_row, idx):
	elem_type = df_row['displayNames']
	bbox = df_row['bboxes']
	inner_text = df_row['predicted text']
	html = f"""<{elem_type} id={idx}>{inner_text}</{elem_type}>"""
	return html

def df_to_html(df):
	s = ''
	for index, row in df.iterrows():
		s += html_from_UIE(row, index) + '\n'
	return s

def bb_intersection_over_minArea(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / min(boxAArea, boxBArea)
	# return the intersection over union value
	return iou

def strls2str(strls):
	s = ''
	for elem in strls:
		s += elem + ' '
	return s[:-1]

def add_text_to_uie(response, ocr):
	conf_threshold = 0
	i = 0

	ids = []
	texts = []
	labels = []
	bboxes = []

	for detection in response["boxes"]:
		if detection["score"] < conf_threshold:
			continue
		text = []
		box = detection["box"]
		x_min, y_min, x_max, y_max = (
			box["topX"],
			box["topY"],
			box["bottomX"],
			box["bottomY"]
		)
		uie_box = [
			x_min * 1280, y_min * 1080, x_max * 1280, y_max * 1080
		]
		for annotation in ocr.text_annotations[1:]:
			top_left = annotation.bounding_poly.vertices[0]
			bottom_right = annotation.bounding_poly.vertices[2]
			ocr_box = [top_left.x, top_left.y, bottom_right.x, bottom_right.y]
			iou = bb_intersection_over_minArea(uie_box, ocr_box)
			if iou > 0.8:
				text.append(annotation.description)   
		text = strls2str(text)

		ids.append(i)
		texts.append(text)
		labels.append(detection["label"])
		bboxes.append([x_min, y_min, x_max, y_max])

		i += 1

	response_df = pd.DataFrame.from_dict({
		"displayNames": labels,
		"bboxes": bboxes,
		"predicted text": texts
	})
	return response_df

def parse_screen():
		print('parsing screen...')
		current_screen = ImageGrab.grab()  # Take the screenshot
		screen_size = current_screen.size
		current_screen = current_screen.resize((RESIZE_WIDTH,RESIZE_HEIGHT))
		current_screen.save('current_screen.png')

		# send screenshot to UIED model to get UIEs
		# print('sending screenshot to tortus UIED model...')
		response = predict_image_object_detection_sample(
			ml_client,
			endpoint_name="uied",
			deployment_name="yolov5",
			path_to_image="current_screen.png"
		)

		# send screenshot to Google OCR to get text
		# print('sending screenshot to google OCR...')
		ocr = detect_text('current_screen.png')

		# merge OCR with UIEs
		# print('merging OCR and UIED...')
		merged_df = add_text_to_uie(response, ocr)
		merged_df.to_csv('uied.csv')
				
		# covert to LLM template format
		# print('converting to LLM template format from dataframe...')
		llm_format = df_to_html(merged_df)
		
		return llm_format

def match_intent(utterance):
	model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

	intent_ls = [
		intents.START_CONSULTATION_NOTE,
		intents.TRANSCRIBE_CONSULTATION,
		intents.SUMMARISE_CONSULTATION,
		intents.PLACE_ORDERS,
		intents.FILE_DIAGNOSES,
		intents.ANSWER_QUESTIONS,
		intents.WRITE_LETTER,
		intents.QUERY_MEDS,
		intents.QUERY_ORDERS,
		intents.LIST_ABILITIES
]

	intent_embeddings = model.encode(intent_ls)
	utterance_embeddings = model.encode(utterance)
	
	cos_scores = cosine_similarity(utterance_embeddings.reshape(1, -1), intent_embeddings)
	cos_scores_torch = torch.from_numpy(cos_scores)
	cos_max = torch.max(cos_scores_torch).item()
	cos_argmax = torch.argmax(cos_scores_torch, dim=1)
	cos = cos_argmax[0].item()

	intent = intent_ls[cos]

	return intent, cos_max

def match_screen(current_screen):
	model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

	screens_ls = [
		epic_screens.PATIENT_LOOKUP,
		epic_screens.SCHEDULE,
		epic_screens.CHART_REVIEW,
		epic_screens.DOCUMENTATION
	]

	screen_labels = ['patient_lookup', 'schedule', 'chart_review', 'documentation']

	epic_embeddings = model.encode(screens_ls)
	screen_embeddings = model.encode(current_screen)

	cos_scores = cosine_similarity(screen_embeddings.reshape(1, -1), epic_embeddings)
	cos_scores_torch = torch.from_numpy(cos_scores)
	cos_max = torch.max(cos_scores_torch).item()
	cos_argmax = torch.argmax(cos_scores_torch, dim=1)
	cos = cos_argmax[0].item()

	intent = screens_ls[cos]
	screen_name = screen_labels[cos]

	return screen_name

 
# Send function
def send():
	msg = "You -> " + e.get()
	txt.insert(END, "\n" + msg)
 
	user = e.get().lower()
	e.delete(0, END)
		
	# Run the rest(user) function asynchronously using a thread
	threading.Thread(target=msg2task, args=(user,)).start()
	# rest(user)

def msg2task(user_msg):
	# match the user command to intents
	intent, score = match_intent(user_msg)
	print(score)

	if float(score) > 0.6:
		# if matched intent is starting a new consult note, attempt extract mrn from user message
		if intent == intents.START_CONSULTATION_NOTE:
			actor.global_mrn = actor.extract_mrn_from_text(user_msg)
			print('mrn: ', actor.global_mrn)
		
		# display matched intent to user
		osler_message = "It looks like you asked me to perform the task: "
		txt.insert(END, "\n" + "OSLER -> " + osler_message + intent)
		# e.delete(0, END)
			
		# perform task
		picovoice_thread.osler_thinking()
		actor.act(intent)

	else:
		# display matched intent to user
		txt.insert(END, "\n" + "OSLER -> This request is not currently supported.")


# GUI
root = Tk()
root.title("OSLER")
 
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
 
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 1080
# DEVICE_SIZE = (1440, 900)
DEVICE_SIZE = (1791, 1119)

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('./tortus-374118-e15fd1ca5b60.json')

ACCESS_KEY = os.environ['PICOVOICE_ACCESS_KEY']

# to prevent the huggingface tokenizer parallelisation error
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set google speech-to-text application credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/christophertan/Desktop/osler1/tortus-374118-e15fd1ca5b60.json"

actor = Actor()

global_mrn = ''

# labe1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="OSLER", font=FONT_BOLD, pady=10, width=20, height=1).grid(
# 	row=0)

img1 = ImageTk.PhotoImage(file="./demo_screenshots/osler_sleep_smaller.png")
label = Label(root, bg=BG_COLOR, image=img1)
label.grid(row=0)
 
txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
txt.grid(row=1, column=0, columnspan=2)
 
scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)
 
e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=55)
e.grid(row=2, column=0)
 
send_button = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
			  command=send).grid(row=2, column=1)

picovoice_thread = PicovoiceThread(label, ACCESS_KEY)
picovoice_thread.start()
while not picovoice_thread.is_ready():
	pass
 
root.mainloop()
# -*- coding: utf-8 -*-
"""hal.py
A path towards machine general intelligence.

"""
##################
# LOAD LIBRARIES #
##################
# Load system libraries
import io
import os
import subprocess
import sys
from pathlib import Path
from contextlib import closing

# Load third-party libraries
import openai
import soundfile as sf
import speech_recognition as sr
from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

############
# AUDITION #
############
# Put temporarily recorded audio for Whisper to decode here.
tmp_path = Path(".tmp.wav")

# Mic input settings
r = sr.Recognizer()
r.energy_threshold = 300
r.pause_threshold = 0.8
r.dynamic_energy_threshold = False


def transcribe_audio(fp):
	"""Transcribe audio from file"""
	command = f"audition/main -m audition/models/ggml-base.en.bin -f {fp} -otxt"
	subprocess.run(command,
	               shell=True,
	               check=True,
	               stdout=subprocess.DEVNULL,
	               stderr=subprocess.STDOUT)
	with open(f"{fp}.txt", "r", encoding="utf-8") as f:
		return f.read().strip()


##########
# SPEECH #
##########
# Configure AWS Polly
session = Session()
polly = session.client("polly")


def tts(text):
	"""Text to speech"""
	# Request speech synthesis
	try:
		polly_response = polly.synthesize_speech(Text=text,
		                                         OutputFormat="mp3",
		                                         VoiceId="Kevin",
		                                         Engine="neural")
	except (BotoCoreError, ClientError) as error:
		print(error)
		sys.exit(-1)

	# Access the audio stream from the response
	if "AudioStream" in polly_response:
		with closing(polly_response["AudioStream"]) as stream:
			output = ".response.mp3"
			try:
				with open(output, "wb") as f:
					f.write(stream.read())
			except IOError as error:
				print(error)
				sys.exit(-1)
			play_command = f"sleep 0.05 && ffplay -nodisp -autoexit {output}"
			subprocess.run(play_command,
			               shell=True,
			               check=True,
			               stdout=subprocess.DEVNULL,
			               stderr=subprocess.STDOUT)
	else:
		print("Could not stream audio")
		sys.exit(-1)


#############
# COGNITION #
#############
# Load environmental variables
load_dotenv()
openai.api_key = os.environ.get("OPENAI_KEY")

# Connect to GPT-3
completion = openai.Completion()

# Initial GPT-3 prompt
chat_prompt = """
The following is a transcript of a conversation between a human and a fully consious machine
intelligence named HAL reliably passes the turing test. HAL always answers questions truthfully and
fully and is fully aware of its own existence.
""".replace("\n", " ").strip()

chat_log = f"""{chat_prompt}

Human: Hello.
HAL: Hello.
"""


def query(question, current_chat_log):
	"""Query GPT-3 with chat log as prompt."""
	prompt = f"{current_chat_log}Human: {question}\nHAL:"
	query_response = completion.create(prompt=prompt,
	                                   engine="davinci",
	                                   stop=["Human:", "Humans:", "HAL:"],
	                                   temperature=0.95,
	                                   top_p=1,
	                                   frequency_penalty=0,
	                                   presence_penalty=0.6,
	                                   best_of=1,
	                                   max_tokens=150)
	answer = query_response.choices[0].text.strip()
	return answer


with sr.Microphone(sample_rate=16000) as source:
	# Get speech and process after breaks
	while True:
		# Record and save audio prompts
		audio = r.listen(source)
		data = io.BytesIO(audio.get_wav_data())
		y, sr = sf.read(data)
		sf.write(tmp_path, y, sr)

		# Transcribe audio to text
		result = transcribe_audio(tmp_path)
		print(result)

		# Handle specific command prompts
		if "terminate" in result.lower():
			tts("Taking all systems offline.")
			break

		# Get response from GPT-3
		response = query(result, chat_log)
		print(">>>", response)

		# Speak response
		tts(response)

		# Append to working memory
		chat_log += f"Human: {result}\nHAL:{response}"

# Remove temporary paths
tmp_path.unlink()
Path(".tmp.wav.txt").unlink()
Path(".response.mp3").unlink()

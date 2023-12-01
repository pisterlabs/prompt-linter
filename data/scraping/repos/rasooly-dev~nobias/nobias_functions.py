import openai
from dotenv import load_dotenv
import os
import json
import re
import requests

import numpy as np
from dotenv import load_dotenv
from hume import HumeBatchClient
from hume.models.config import LanguageConfig
from pathlib import Path
from typing import Any, Dict, List

from .positivenegative import *
from .political import *

load_dotenv()
hume_api_key = os.getenv("HUME_API_KEY")
print(f"\n\n\n{hume_api_key}\n\n\n")
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}"
	}

def main():
	print("Hello World")
	print(test_generation("CalHacks"))

def test_generation(prompt: str):
	test_prompt = f"Write a sentence about {prompt}"

	data = {
		"model": "gpt-4",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": test_prompt}
		],
		"max_tokens": 50
	}

	response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

	if response.status_code == 200:
		return response.json()["choices"][0]["message"]["content"]
	else:
		raise Exception(f"Error {response.status_code}: {response.text}")


def generate_info(text: str):
	prompt = f"Given a piece of text, please generate a concise and unbiased summary of the content." \
		 f" Additionally, extract three statements from the text that can be fact-checked, and a good title." \
		 f" Please provide the response in JSON format." \
		 f"\n\nExample response format:" \
		 f"\n{{" \
		 f"\n    \"summary\": \"This is a summary of the text.\"," \
	 	 f"\n    \"title\": \"This is a good title of the text.\"," \
		 f"\n    \"facts\": [" \
		 f"\n        \"Fact 1 statement.\"," \
		 f"\n        \"Fact 2 statement.\"," \
		 f"\n        \"Fact 3 statement.\"" \
		 f"\n    ]" \
		 f"\n}}" \
		 f"\n\nHere is the text to summarize and extract facts from:\n{text}"
	
	data = {
		"model": "gpt-3.5-turbo-16k",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}
		],
		"max_tokens": 5000
	}

	response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

	if response.status_code == 200:
		resp = response.json()["choices"][0]["message"]["content"]
		json_resp = json.loads(resp.replace('\n', ' '))
		return json_resp
	else:
		raise Exception(f"Error {response.status_code}: {response.text}")

def generate_neutral_article(text: str):
	prompt = f"Given a piece of text, please generate a concise and unbiased one paragraph article using the content of the same length." \
			f" Avoid overly negative words."\
			f" Also generate a title for the new unbiased objective article."\
			f" Return the article in JSON format parsable by the json.loads() function in python. Example below:\n"\
			f"{{\"title\":\"example title\", \"text\":\"Example Unbiased Article\"}}"\
			f"\n\nHere is the text to generate the unbiased article from:\n{text}"
	
	data = {
		"model": "gpt-4",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}
		],
		"max_tokens": 5000
	}

	response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

	if response.status_code == 200:
		resp = response.json()["choices"][0]["message"]["content"]
		resp = resp.replace('\n', ' ')
		json_resp = json.loads(resp)
		return json_resp
	else:
		raise Exception(f"Error {response.status_code}: {response.text}")
	
def generate_positive_article(text: str):
	prompt = f"Given a piece of text, please generate a concise one-paragraph article using the content of the same length." \
			f" Ensure the article portrays a positive and understanding perspective." \
			f" Also, generate a title for the new article that highlights the positive aspects." \
			f" Return the article in JSON format parsable by the json.loads() function in python. Example below:\n" \
			f"{{\"title\":\"example title\", \"text\":\"Example Unbiased Article\"}}\"" \
			f"\n\nHere is the text to generate the unbiased article from:\n{text}"
	
	data = {
		"model": "gpt-4",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}
		],
		"max_tokens": 5000
	}

	response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

	if response.status_code == 200:
		resp = response.json()["choices"][0]["message"]["content"]
		resp = resp.replace('\n', ' ')
		json_resp = json.loads(resp)
		return json_resp
	else:
		raise Exception(f"Error {response.status_code}: {response.text}")
	
def generate_negative_article(text: str):
	prompt = f"Given a piece of text, please generate a concise article using the content of the same length." \
		 f" Provide a negative perspective that acknowledges flaws influencing the topic." \
		 f" Offer negative viewpoints critiquing the subject" \
		 f" Also, generate a title for the new article that highlights a nuanced viewpoint." \
		 f" Return the article in a JSON format. Example below:\n" \
		 f"{{\"title\":\"example title\", \"text\":\"Example Unbiased Article\"}}\"" \
		 f"\n\nHere is the text to generate the article from:\n{text}"

	data = {
		"model": "gpt-4",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}
		],
		"max_tokens": 1000
	}

	response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

	if response.status_code == 200:
		resp = response.json()["choices"][0]["message"]["content"]
		print(resp)
		resp = resp.replace('\n', ' ')
		print('\n\n\n')
		print(resp)
		json_resp = json.loads(resp, strict=False)
		return json_resp
	else:
		raise Exception(f"Error {response.status_code}: {response.text}")
	

	## start generate emotion function
def generate_emotions(text: str):

	TEXT = text
	file_path = "outtext.txt"

	with open(file_path, "w") as fp:
			fp.write(TEXT)

	client = HumeBatchClient(hume_api_key)
	config = LanguageConfig(granularity="conversational_turn")
	job = client.submit_job(None, [config], files=[file_path])

	print("running...", job)

	job.await_complete()
	print("Job completed with status: ", job.get_status)

	emotion_embeddings = []
	full_predictions = job.get_predictions()
	for source in full_predictions:
		predictions = source["results"]["predictions"]
		for prediction in predictions:
			language_predictions = prediction["models"]["language"]["grouped_predictions"]
			for language_prediction in language_predictions:
				for chunk in language_prediction["predictions"]:
					emotion_embeddings.append(chunk["emotions"])
				
	class Stringifier:
		RANGES = [(0.26, 0.35), (0.35, 0.44), (0.44, 0.53), (0.53, 0.62), (0.62, 0.71), (0.71, 10)]
		ADVERBS = ["slightly", "somewhat", "moderately", "quite", "very", "extremely"]
	
		ADJECTIVES_48 = [
			"admiring", "adoring", "appreciative", "amused", "angry", "anxious", "awestruck", "uncomfortable", "bored",
			"calm", "focused", "contemplative", "confused", "contemptuous", "content", "hungry", "determined",
			"disappointed", "disgusted", "distressed", "doubtful", "euphoric", "embarrassed", "disturbed", "entranced",
			"envious", "excited", "fearful", "guilty", "horrified", "interested", "happy", "enamored", "nostalgic",
			"pained", "proud", "inspired", "relieved", "smitten", "sad", "satisfied", "desirous", "ashamed",
			"negatively surprised", "positively surprised", "sympathetic", "tired", "triumphant"
		]
	
		ADJECTIVES_53 = [
			"admiring", "adoring", "appreciative", "amused", "angry", "annoyed", "anxious", "awestruck", "uncomfortable",
			"bored", "calm", "focused", "contemplative", "confused", "contemptuous", "content", "hungry", "desirous",
			"determined", "disappointed", "disapproving", "disgusted", "distressed", "doubtful", "euphoric", "embarrassed",
			"disturbed", "enthusiastic", "entranced", "envious", "excited", "fearful", "grateful", "guilty", "horrified",
			"interested", "happy", "enamored", "nostalgic", "pained", "proud", "inspired", "relieved", "smitten", "sad",
			"satisfied", "desirous", "ashamed", "negatively surprised", "positively surprised", "sympathetic", "tired",
			"triumphant"
		]
	
		@classmethod
		def scores_to_text(cls, emotion_scores: List[float]) -> str:
			if len(emotion_scores) == 48:
				adjectives = cls.ADJECTIVES_48
			elif len(emotion_scores) == 53:
				adjectives = cls.ADJECTIVES_53
			else:
				raise ValueError(f"Invalid length for emotion_scores {len(emotion_scores)}")
	
	
			# Return "neutral" if no emotions rate highly
			if all(emotion_score < cls.RANGES[0][0] for emotion_score in emotion_scores):
				return "neutral"
	
			# Construct phrases for all emotions that rate highly enough
			phrases = [""] * len(emotion_scores)
			for range_idx, (range_min, range_max) in enumerate(cls.RANGES):
				for emotion_idx, emotion_score in enumerate(emotion_scores):
					if range_min < emotion_score < range_max:
						phrases[emotion_idx] = f"{cls.ADVERBS[range_idx]} {adjectives[emotion_idx]}"
	
			# Sort phrases by score
			sorted_indices = np.argsort(emotion_scores)[::-1]
			phrases = [phrases[i] for i in sorted_indices if phrases[i] != ""]
	
			# If there is only one phrase that rates highly, return it
			if len(phrases) == 0:
				return phrases[0]
	
	def get_top_emotions(emotions_list):
		sorted_emotions = sorted(emotions_list[0], key=lambda x: x["score"], reverse=True)
		top_emotions = sorted_emotions[:5]
		return top_emotions
	
	return(get_top_emotions(emotion_embeddings))

def get_context_score(text: str):
	prompt = f"Provide a single integer score from 0 to 15, where 0" \
		f"represents extreme bias and 25 represents utmost neutrality. Provide nothing else, just a SINGLE INTEGER." \
		f"For example: 25" \
		f"\n\nHere is the text to generate the article from:\n{text}"

	data = {
		"model": "gpt-4",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}
		],
		"max_tokens": 50
	}

	response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

	if response.status_code == 200:
		resp = response.json()["choices"][0]["message"]["content"]
		return resp
	else:
		raise Exception(f"Error {response.status_code}: {response.text}")

def generate_score(text: str):
	"""
	Rubric (out of 50): 
	25 pts: political bias
	25 pts: context (explanation on why)
	"""

	pos_neg_res = positiveNegative(text)
	political_res = politicalAffiliation(text)
	context_res = get_context_score(text)

	# TODO: fix neutrality, ask emir
	def neutrality_analysis():
		spectrum_val = pos_neg_res[0]

		neutrality_score = 20 - abs(spectrum_val - 50) * 20 / 50
		return neutrality_score

	def political_bias_analysis():
		spectrum_val = political_res[0]
		bias_score = 25 - abs(spectrum_val) * 25 / 100
		return bias_score
	
	pos_neg_score = int(neutrality_analysis())
	political_score = int(political_bias_analysis())
	context_score = int(context_res)

	final_score = political_score + context_score
	return final_score

	
def generate_in_depth_analysis(text: str):
	prompt = f"Please generate an in depth analysis about the article given" \
		f"Make it analytical, but not too extremely long, maybe a nice paragraph." \
		f"\n\nHere is the text to generate the article from:\n{text}"

	data = {
		"model": "gpt-4",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt}
		],
		"max_tokens": 5000
	}

	response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

	if response.status_code == 200:
		resp = response.json()["choices"][0]["message"]["content"]
		return resp
	else:
		raise Exception(f"Error {response.status_code}: {response.text}")

	
	

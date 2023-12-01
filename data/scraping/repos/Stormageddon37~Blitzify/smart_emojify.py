import random

import emojis
import openai
from decouple import config

from translate import text_to_english

openai.api_key = config('OPENAI_API_KEY')


def smart_emojify_text(text: str) -> str:
	response = openai.Completion.create(
		model="text-davinci-002",
		prompt=f"Convert text into emojis. {text}:",
		temperature=1.0,
		max_tokens=70,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0.85,
		stop=["\n"]
	)
	return ''.join(emojis.get(response.choices[0].get('text'))).rstrip()


def slow_smart_emojify_text(text: str, percentage: int) -> str:
	words = text.split()
	for i, word in enumerate(words):
		x = random.randint(1, 100)
		if x <= percentage:
			english_word = text_to_english(word).lower()
			emoji_word = smart_emojify_text(english_word)
			words[i] = f' {word} {emoji_word}'
	return ''.join(words).rstrip()

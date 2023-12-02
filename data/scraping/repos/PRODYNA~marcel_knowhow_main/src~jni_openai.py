import os

import openai
from dotenv import load_dotenv

API_KEY_NAME = "openai.api_key"

JSON_EXAMPLE = """
[
	{
		"id": 1,
		"question": "Did the French Revolution start in the 18th Century?",
		"yes_answer": true
	},
	{
		"id": 2,
		"question": "Did World War II end in 1942?",
		"yes_answer": false
	}
]
"""


class Message:
	def __init__(self, role: str, content: str):
		self.role = role
		self.content = content
	

class Choice:
	def __init__(self, finish_reason: str, index: int, message: Message):
		self.finish_reason = finish_reason
		self.index = index
		self.message = message 


class OpenAi:
	
	def __init__(self) -> None:
		load_dotenv()
		key_value = os.environ.get(API_KEY_NAME)
		if key_value is None:
			raise Exception(f'"{API_KEY_NAME}" is not set in .env file!')
		openai.api_key = key_value

	def say_hello(self) -> None:
		person_name = "Jens"

		prompt = f'Create a simple greeting for a user named "{person_name}". \
			Answer only with the greeting.'

		response = openai.ChatCompletion.create(
			model="gpt-4",
			messages=[
				{"role": "user", "content": prompt},
			]
		)

		choice: Choice = response["choices"][0]  # type: ignore
		message = choice.message
		greeting = message.content
		print(f"Greeting: {greeting}")
	
	def create_quizz_questions(self, no_questions: int) -> str:
		prompt = f"""Create {no_questions} questions based on European history which are
'general knowledge and can be answered with a "yes" or "no". Answer only in JSON format.
Here is an example:
{JSON_EXAMPLE}
---
Answer only in the requested format without any comments or descriptions.'
"""

		print("Contacting OpenAI API. This may take a few minutes...")
		response = openai.ChatCompletion.create(
			model="gpt-4",
			messages=[
				{"role": "user", "content": prompt},
			]
		)

		choice: Choice = response["choices"][0]  # type: ignore
		message = choice.message
		quizz_questions = message.content
		print("Response received!")
		return quizz_questions


def main() -> None:
	open_ai = OpenAi()
	result = open_ai.create_quizz_questions(10)
	print(f'Result: "{result}"')
	

if __name__ == "__main__":
	main()

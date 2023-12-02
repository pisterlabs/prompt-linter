# import library
import os
import openai

# configure openai to your account 
from dotenv import load_dotenv, find_dotenv

#from summarizer.llama2_summarizer import Llama
from generator.joke_type import JokeType, completion_prompt
from generator.joke_examples import q_and_a_examples, knock_knock_examples
# finds and loads the .env file
load_dotenv(find_dotenv())

# call the variable from the loaded file
openai.api_key = os.environ.get("openai_api_key")


knockknock = JokeType(
    joke_type="knock knock",
	unstructured_joke_list= knock_knock_examples
)

q_and_a = JokeType(
    joke_type="Q and A",
	unstructured_joke_list= q_and_a_examples
)

#API Request Example
# {
# "text": "bunny", 
# "type": "Knock Knock"
# }
def generate_joke(text, type):

	if type == "Knock Knock":
		prompt, joke_ids = knockknock.n_shot_prompt(text, amount_of_examples=5)
	elif type == "Q and A":
		prompt, joke_ids = q_and_a.n_shot_prompt(text, amount_of_examples=3)
	elif type == "Completion":
		prompt = completion_prompt(text)

	completion = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages = prompt
	)

	return completion.choices[0].message.content, joke_ids

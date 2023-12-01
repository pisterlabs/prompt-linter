import openai
import os
import sys

######################
# API KEY GOES HERE
######################

openai.api_key = ""

######################
# API KEY GOES ABOVE
######################

def ask_chatng(prompt):
	# Set the model
	model = "text-davinci-003"

	# Make the request to the text-davinci-003 API, because it doesn't use ChatGPT
	response = openai.Completion.create(engine=model, \
		prompt=prompt, max_tokens=1024, temperature=0.5, \
		top_p=1, frequency_penalty=0, presence_penalty=0)

	# Construct the response
	output = response['choices'][0]['text']

	return output[2:]

def query(prompt):
	try:
		question = prompt
		answer = ask_chatng(question)
		return answer

	except Exception as e:
		print(repr(e))

if __name__ == "__main__":
	query(prompt)

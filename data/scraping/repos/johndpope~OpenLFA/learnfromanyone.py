import openai
from kgptj import SimpleCompletion

class Person:
	# the name of the person
	name = ""
	
	# message cache are the previous messages that are sent to the
	# ai as training data, enabling it to carry on conversations and whatnot
	messagecachelen = 5
	messagecache = []

	# this is a local log of the "person"'s message history, useful for exporting logs
	messagelog = []

	# if this is set, it will use the openai GPT-3, if not, it will fall back on GPT-J
	openai_key = ""

	def __init__(self, name):
		# provides the AI context for the conversation and who its supposed to be
		self.stem = f"A conversation with {name}\n\nInterviewer: \"What is your name?\"\n{name}: \"My name is {name}\"\n" 
		self.name = name
	
	def gpt3_respond(self, question):
		prepend = self.stem + "\n".join(self.messagecache)

		prompt = f"{prepend}\nQ: \"{question}\"\nA: \""

		response = openai.Completion.create(
			engine="davinci", 
			prompt=prompt, 
			max_tokens=80, 
			temperature=0.75, 
			stop="\"",
			frequency_penalty=0.5,
			)

		response_text = response["choices"][0]["text"]

		self.messagecache.append(f"Interviewer: \"{question}\"\n{self.name}: \"{response_text}\"\n")
		self.messagelog.append(f"Interviewer: \"{question}\"\n{self.name}: \"{response_text}\"\n")

		self.messagecache = self.messagecache[::-1][0:self.messagecachelen]

		return response_text

	def gptj_respond(self, question):

		prepend = self.stem + "\n".join(self.messagecache)

		prompt = f"{prepend}\nQ: \"{question}\"\nA: \""

		query = SimpleCompletion(
			prompt, 
			length=75, 
			t=0.78, 
			)

		response_text = query.simple_completion().split("\"")[0]

		self.messagecache.append(f"Interviewer: \"{question}\"\n{self.name}: \"{response_text}\"\n")
		self.messagelog.append(f"Interviewer: \"{question}\"\n{self.name}: \"{response_text}\"\n")

		self.messagecache = self.messagecache[::-1][0:self.messagecachelen]

		return response_text
	
	def respond(self, statement):
		if self.openai_key:
			openai.api_key = self.openai_key
			return self.gpt3_respond(statement)
		else:
			return self.gptj_respond(statement)
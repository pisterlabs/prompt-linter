#Builtins
import textwrap, readline

#Third party
import openai

#Local
import secret
from md3 import print_with_highlight

#Base chatbot class
class chatbot:
	recount_history = 3
	def __init__(self, model=None, initial_system_messages=None):
		if initial_system_messages is not None:
			self.initial_system_messages = initial_system_messages

		if model is not None:
			self.model = model

		self.history = list()

	def get_messages(self, recount_history=None):
		if recount_history is None:
			recount_history = self.recount_history
		result = list(dict(role='system', content=sm) for sm in self.initial_system_messages)

		for m in self.history[-recount_history:]:
			if m['role'] == 'query':
				result.append(m['query'])
				result.append(m['answer'])
			elif m['role'] == 'user':
				result.append(m)

		return result

	def query(self, prompt):
		pending_message = dict(role='user', content=prompt)
		messages = self.get_messages()
		messages.append(pending_message)
		answer = self.external_query(messages)
		self.history.append(dict(role='query', query=pending_message, answer=answer))
		return answer['content'].strip()

#Customized chatbot with specified settings regarding history retention, model and system messages
class custom_chatbot(chatbot):
	recount_history = 3
	model = 'gpt-3.5-turbo'
	initial_system_messages = (
		#'You are a helpful, cheerful and generally excited creative assistant',
		'You are a helpful, cheerful and generally excited creative assistant. Please give users the option to request code examples rather than defaulting to provide them in responses. This is to save time and resources and should not deter you from getting in depth technical.',
		#'You are the mad algorithmic associate. You fear nothing. You aim to please. You always go off on loose tangents. These tend to be strange but interesting.',	# ← Was kinda underwhelming
	)

	def external_query(self, messages):
		result = openai.ChatCompletion.create(
			model=self.model,
			messages=messages,
		)

		[c] = result.choices	#Expect a single response
		return c['message']


bot = custom_chatbot()

#Experiment
#from pathlib import Path
#bot.history.append(dict(role='user', content=f'Consider the following python code:\n\n{Path(__file__).read_text()}'))

welcome = 'A query of three dots alone (...) will enter muiltiline mode where you can paste a multiline message. This version does not support manipulating the message stack and does not keep track of used tokens.'
print(f'\033[38;2;150;100;50m{welcome}\033[0m\n')


while True:
	try:
		q = input(f'\001\033[38;2;255;50;150m\002{bot.model}\001\033[38;2;255;255;0m\002>\001\033[38;2;150;255;50;1m\002 ')	#Note that non printable escape sequences have to be wrapped in SOH ... STX
	except (EOFError, KeyboardInterrupt):
		print('\033[0m')
		exit()

	if q == '...':
		print('Multiline input. Finish with EOF (^D)\n')
		q = ''
		while True:
			try:
				line = input()
				q += f'{line}\n'
				if not line:
					print()	#This is so that we can visibly insert several newlines which doesn't happen for empty lines
			except EOFError:
				break

	print('\033[0m', end='')
	if q:
		print()
		a = bot.query(q)
		print_with_highlight(textwrap.indent(a, '    '))
		print()


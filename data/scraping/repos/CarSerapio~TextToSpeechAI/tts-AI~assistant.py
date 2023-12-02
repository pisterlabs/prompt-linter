import openai 
from elevenlabs import generate, play, set_api_key 

with open("token.txt") as file: 
	tokens = file.readlines() 

openai.api_key = tokens[0].rstrip() 
set_api_key(tokens[1].rstrip()) 

class Assistant():

	def __init__(self): 
		self.messages = [
			{"role": "system", "content": "You are a helpful assistant."}
		]
	
	def AssistantResponse(self, user_text):
		self.user_text = user_text

		while True:

			# if user says stop, then break the loop
			if self.user_text == "stop":
				break
			
			# storing the user message in the messages list
			self.messages.append({"role": "user", "content": self.user_text})

			# getting the response from OpenAI API
			response= openai.ChatCompletion.create(
				model="gpt-3.5-turbo",
				messages=self.messages
			)

			# appending the generated response so that AI remembers past responses
			self.messages.append({"role":"assistant", "content":str(response['choices'][0]['message']['content'])})
			
			# generate audio 

			audio = generate(
				  text=str(response['choices'][0]['message']['content']),
				  voice="Josh",
				  model="eleven_monolingual_v1"
				  )

			# returning the response
			play(audio) 

			return response['choices'][0]['message']['content']

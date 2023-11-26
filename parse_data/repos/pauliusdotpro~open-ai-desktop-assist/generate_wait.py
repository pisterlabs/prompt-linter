from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
language = os.getenv('LANGUAGE') 
api_key = os.getenv('API_KEY')
voice = os.getenv('VOICE')

client = OpenAI(api_key=api_key)

def text_to_speech(text, output_file="output.mp3"):
	response = client.audio.speech.create(
		model="tts-1",
		voice=voice,
		input=text,
	)

	response.stream_to_file(output_file)

#text_to_speech("Aha, alright, let me think for a second.", "thinking.mp3")
#text_to_speech("I think I've got it!", "got_it.mp3")
#text_to_speech("Hello, I am your personal assistant. How can I help you?", "hello.mp3")
text_to_speech("I'm sorry, I didn't quite understand that. Could you repeat that?", "repeat.mp3")
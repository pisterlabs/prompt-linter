import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Engine.list()

def generate_prompt():
	return """2 * 4 ="""

kwargs = {
	"temperature":0.5,
  	"max_tokens":30,
  	"top_p":1,
  	"stop":["\n"],
  	"frequency_penalty":0.5,
  	"presence_penalty":0.0
}

response = openai.Completion.create(
		engine="code-davinci-001",
		prompt=generate_prompt(), 
		temperature=0.5,
		max_tokens=30,
		top_p=1,
		stop=["\n"],
		frequency_penalty=0.5,
		presence_penalty=0.0
	)

output = response.choices[0].text
print(output)

  
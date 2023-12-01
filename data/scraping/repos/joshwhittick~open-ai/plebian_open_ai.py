import openai
from api_key import api_data
openai.api_key=api_data

completion=openai.Completion()

def Reply(question):
	prompt=f'Josh: {question}\nOpenAI:'
	response=completion.create(
		prompt=prompt,
		engine="text-davinci-002",
		max_tokens=300,
		top_p=1,
		presence_penalty=0.6)
	answer=response.choices[0].text.strip()
	return answer

while True:
	ans = Reply(input("Enter prompt for OpenAI:" ))
	print(f'OpenAI:{ans}')
	

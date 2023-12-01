import openai
import getkey as k
openai.api_key = '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'
messages = [ {"role": "system", "content":
			"You are a intelligent assistant."} ]
while True:
	message = input("User : ")
	if message:
		messages.append(
			{"role": "user", "content": message},
		)
		chat = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", messages=messages
		)
	reply = chat.choices[0].message.content
	print(f"ChatGPT: {reply}")
	messages.append({"role": "assistant", "content": reply})

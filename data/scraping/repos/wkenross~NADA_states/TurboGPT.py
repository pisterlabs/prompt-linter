import openai
import pyperclip

# Use davinci and davinci-codex for better results
# instead of original gpt-3.5-turbo

openai.api_key = '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'
messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]
while True:
	print("\033[38;2;38;139;210m", end="")  # blue color
	message = input("User : ")
	print("\033[0m", end="")  # reset color
	if message:
		messages.append(
			{"role": "user", "content": message},
		)
		chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages
		)
	reply = chat.choices[0].message.content
	print("\033[33;2;173;203;0mReply:", end=" ")  # green color	
	print(f"ChatGPT: {reply}")
	messages.append({"role": "assistant", "content": reply})
	print("\033[0m", end="")  # reset color

	# Copy the content in reply
	pyperclip.copy(reply)
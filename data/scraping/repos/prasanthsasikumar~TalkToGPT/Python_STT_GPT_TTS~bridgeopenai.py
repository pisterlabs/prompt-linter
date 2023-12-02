#openai.api_key = "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"
import openai

openai.api_key = "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"
messages = [
 {"role": "system", "content" : "You’re a kind helpful assistant"}
]

while True:
    content = input("User: ")
    messages.append({"role": "user", "content": content})
    
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )

    chat_response = completion.choices[0].message.content
    print(f'ChatGPT: {chat_response}')
    messages.append({"role": "assistant", "content": chat_response})


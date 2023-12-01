import openai

response = openai.Completion.create(model="text-davinci-002", prompt="Say this is a test", max_tokens=6, temperature=0)

print(response)

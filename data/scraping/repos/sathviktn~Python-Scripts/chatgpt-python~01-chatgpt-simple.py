import openai

openai.api_key = "### YOUR API KEY ###"

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Give me 3 ideas for apps I could build with openai apis "}])
print(completion.choices[0].message.content)

# Requires paid chatgpt account
# Find more at: https://github.com/AIAdvantage/chatgpt-api-youtube
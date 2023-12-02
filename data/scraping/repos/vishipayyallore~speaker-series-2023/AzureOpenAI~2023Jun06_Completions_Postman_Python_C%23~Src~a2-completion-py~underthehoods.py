import os
import openai

openai.api_type = "azure"
openai.api_base = "https://azure-openai-dev-001.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

user_prompt = "Write a small tweet for twitter; on my Dosa Center. 1001 varities, home made spices, butter, and lots of love. I need 5 tweets."

# user_prompt = "gpt-35-turbo-dev-001"
# engine="text-davinci-003-dev-0109",

response = openai.Completion.create(
  engine="gpt-35-turbo-dev-001",
  prompt=user_prompt,
  temperature=1,
  max_tokens=150,
  top_p=0.5,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

# print('Created: ', response['created'], 'Choices: ', response['choices'])

# print(response)

print(response.choices[0].text)
# print(response.choices[1].text)
# print(response.choices[2].text)
# print(response.choices[3].text)
# print(response.choices[4].text)

# print(type(response))


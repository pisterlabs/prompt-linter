import openai
from dotenv import dotenv_values

config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

prompt = """
      Tell me a funny but inspiration story no more than 200 words to get me motivated
   """
response = openai.Completion.create(
    prompt=prompt,
    model="text-davinci-003",
    max_tokens=200,
    temperature=0
)

print(response.choices[0].text)
import os
import openai

openai_api_key = os.environ.get('OPENAI_API_KEY')

query_string = "Who is the better actor between Tom Hiddleston or Scarlett Johansson"
comparision_prompt = f"Is this question comparing two people ? : {query_string}"
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=comparision_prompt,
  temperature=0,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)

print(f'1.{response["choices"][0]["text"]}')


import openai
openai.api_key = ''

response = openai.Completion.create(
  model="text-davinci-003",
  temperature=0.5,
  max_tokens=100,
  prompt="请给我的花店起个名")

print(response.choices[0].text.strip())

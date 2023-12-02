import openai

openai.api_key = 'your-api-key'

response = openai.Completion.create(
  engine="text-davinci-003.5",  # or "text-davinci-004" for GPT-4
  prompt="Translate the following English text to French: '{}'",
  max_tokens=60
)

print(response.choices[0].text.strip())

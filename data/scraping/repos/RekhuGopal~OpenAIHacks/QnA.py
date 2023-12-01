import os
import openai

os.environ["OPENAI_API_KEY"] = "Your Open AI API Key"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Q: who is elon musk?\n A:",
  temperature=0,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)

#print(response)
print(response['choices'][0]['text'])
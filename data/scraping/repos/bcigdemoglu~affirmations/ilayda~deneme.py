import os
import openai

# Assign variables with user input
input = input(" What are you currently dealing with? ")
prompt = f"write an affirmation for me as I am dealing with {input}."
print (prompt) # Send this to OpenAI 

# Generate text with GPT-3
openai.api_key = os.getenv("CLOSE_API_KEY")
response = openai.Completion.create(
  model="text-davinci-002",
  prompt=prompt,
  temperature=0.7,
  max_tokens=48,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response)


#
# BrightonSEO April 2023, Basic GPT example.
#

import openai
openai.api_key = "***key here***"

# Build the completion
completion = openai.ChatCompletion.create(
  model = 'gpt-3.5-turbo',
  messages = [
     {"role": "user", "content": "Write a summary of the history of the BrightonSEO event."}
  ],
  temperature = 0
)

# Place our content into the variable aiResponse
aiResponse = completion['choices'][0]['message']['content']

# Print the contents of aiResponse to screen
print(aiResponse)

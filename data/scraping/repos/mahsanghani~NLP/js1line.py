import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Use list comprehension to convert this into one line of JavaScript:\n\ndogs.forEach((dog) => {\n    car.push(dog);\n});\n\nJavaScript one line version:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=[";"]
)

# Prompt
# Use list comprehension to convert this into one line of JavaScript:

# dogs.forEach((dog) => {
#     car.push(dog);
# });

# JavaScript one line version:
# Sample response
# [dogs.forEach(dog => car.push(dog))]
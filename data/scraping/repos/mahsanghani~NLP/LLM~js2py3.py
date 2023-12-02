import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="#JavaScript to Python:\nJavaScript: \ndogs = [\"bill\", \"joe\", \"carl\"]\ncar = []\ndogs.forEach((dog) {\n    car.push(dog);\n});\n\nPython:",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# #JavaScript to Python:
# JavaScript: 
# dogs = ["bill", "joe", "carl"]
# car = []
# dogs.forEach((dog) {
#     car.push(dog);
# });

# Python:
# Sample response
# dogs = ["bill", "joe", "carl"]
# car = []
# for dog in dogs:
#     car.append(dog)

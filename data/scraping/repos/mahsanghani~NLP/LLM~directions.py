import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Create a numbered list of turn-by-turn directions from this text: \n\nGo south on 95 until you hit Sunrise boulevard then take it east to us 1 and head south. Tom Jenkins bbq will be on the left after several miles.",
  temperature=0.3,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Create a numbered list of turn-by-turn directions from this text: 

# Go south on 95 until you hit Sunrise boulevard then take it east to us 1 and head south. Tom Jenkins bbq will be on the left after several miles.
# Sample response
# 1. Go south on 95 
# 2. Take Sunrise Boulevard east 
# 3. Head south on US 1 
# 4. Tom Jenkins BBQ will be on the left after several miles
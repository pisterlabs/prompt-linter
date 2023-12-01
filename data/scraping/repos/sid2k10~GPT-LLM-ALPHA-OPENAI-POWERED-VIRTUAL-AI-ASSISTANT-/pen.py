# Import necessary libraries and modules
import os
import openai

# Import API key from a configuration file (assuming 'coinfig.py' contains the API key)
from coinfig import apikey

# Set the OpenAI API key using the imported key
openai.api_key = apikey

# Create a completion request for OpenAI
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="write a tagline for an ice-cream shop",
    temperature=1,            # Adjust the randomness of the response
    max_tokens=256,            # Set the maximum length of the response
    top_p=1,                   # Adjust the nucleus sampling parameter
    frequency_penalty=0,       # Adjust the response frequency penalty
    presence_penalty=0         # Adjust the response presence penalty
)

# Print or use the generated response
print(response.choices[0].text)


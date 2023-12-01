# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:05:28 2023

@author: Matthew Tavares
"""

import openai

# Set your API key
api_key = "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"

# Initialize the OpenAI API client
openai.api_key = api_key

# Accept user input (the word)
user_input = input("Enter a word: ")


generalEffects = "what is the general pacing, transitions, color correction, and effects of" + user_input + " videos. give specfic  and generalized 1-3 word response for each in the form - (eg. pacing: fast, color correction: warm, effects: slow motion)"
additionalResources = "in general, for " + user_input + ", do they need stock images to complement current clips? Answer yes/no 1 word. Also, for " + user_input + ", in general, would it benefit from AI generated summary voice over? Answer yes/no in only 1 word response."


# Generate text using GPT-3.5
response = openai.Completion.create(
    engine="text-davinci-002",  # Choose an appropriate engine
    prompt=f"Generate a text document based on the word: {generalEffects} {additionalResources}",
    max_tokens=100,  # Adjust as needed
)

# Extract the generated text
generated_text = response.choices[0].text

# Save the generated text to a text document
with open("outputEditAI.txt", "w") as file:
    file.write(generated_text)

print(generated_text)

import streamlit as st
import openai

API = "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"




# Define OpenAI API key 
openai.api_key = API

# Set up the model and prompt
model_engine = "text-davinci-003"
prompt = "Once upon a time, in a land far, far away, there was a princess who..."

# Generate a response
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

response = completion.choices[0].text
print(response)
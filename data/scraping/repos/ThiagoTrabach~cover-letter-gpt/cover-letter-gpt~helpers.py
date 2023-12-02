import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_KEY']

def gpt_get_completion(prompt, model="gpt-3.5-turbo"): # Andrew mentioned that the prompt/ completion paradigm is preferable for this class
    messages = [{"role": "user", "content": prompt}] 
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response


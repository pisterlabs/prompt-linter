import openai
from dotenv import load_dotenv, find_dotenv
from langchain_reader import langchain_pdf_reader
import os

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

def get_completion_from_messages(messages,
                                model="gpt-3.5-turbo",
                                temperature=0,
                                max_tokens=500):

    response = openai.ChatCompletion.create(model=model,
                                            messages=messages,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
    return response.choices[0].message
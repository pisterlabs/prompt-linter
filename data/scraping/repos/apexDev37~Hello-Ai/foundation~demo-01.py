"""
Demo of Langchain components: models, prompts, and parsers.

written by:   Eugene M.
              https://github.com/apexDev37

date:         nov-2023

usage:        simple demo of using a prompt template to prompt an LLM.
"""

import datetime
import openai

from os import environ as env
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


CHATGPT_MODEL: str = 'gpt-3.5-turbo'

def main() -> None:
  """ Script entry-point func. """

  # Load your Open AI, API key
  # If you don't have one, see:
  # https://platform.openai.com/account/api-keys
  ENV_FILE = find_dotenv()
  if ENV_FILE:
    load_dotenv()

  openai.api_key = env.get('OPENAI_API_KEY')

  # Select LLM model
  # handle deprecation of model
  current_date = datetime.datetime.now().date()
  target_date = datetime.date(2024, 6, 12)

  if current_date > target_date:
    llm_model = CHATGPT_MODEL
  else:
    llm_model = CHATGPT_MODEL + '-0301'

  # Initialize model: Open AI's GPT LLM
  chat = ChatOpenAI(temperature=0.0, model=llm_model,)

  # Ensure prompt consists of the following:
  # desired action and precise context
  greeting = 'hello'
  template = """
    Determine if the provided text delimited in backticks is a common and
    formal greeting. If it is a greeting, determine the language of the provided
    greeting, three synonyms for the given text, and five countries that it is
    most commonly used. Else, recommend three common and formal greetings used around
    the world at random.
    text: `{customer_greeting}`
  """

  prompt_template = ChatPromptTemplate.from_template(template)
  messages = prompt_template.format_messages(customer_greeting=greeting)

  response = chat(messages)
  print(response.content)


if __name__ == '__main__':
  main()

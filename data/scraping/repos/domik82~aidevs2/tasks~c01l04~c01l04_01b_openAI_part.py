import json

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from common.logger_setup import configure_logger

# Task: please write blog post for the provided outline
# Input: 'Wstęp: kilka słów na temat historii pizzy', 'Niezbędne składniki na pizzę', 'Robienie pizzy',
# 'Pieczenie pizzy w piekarniku'


system_template = """
As a blogger your task is to write blog post for the outlines provided by user.
Each section of the blogpost is based on outline. Single outline is encapsulated in single quotes 
All outlines are separated by coma. Prepare short paragraph for each outline.  
Provide generated content in JSON format as an array of elements.

The blog post needs to be in polish.
Generate only 3 sentences per single outline.

###
Sample of generated input:
'Wstęp: kilka słów na temat historii pizzy', 'Niezbędne składniki na pizzę'
###
Sample of generated content:
[
  {{
    "section": "Wstęp: kilka słów na temat historii pizzy",
    "content": "Pizza jest jednym z najbardziej popularnych dań na całym świecie...."
  }},
  {{
    "section": "Niezbędne składniki na pizzę",
    "content": "Aby przygotować pyszną pizzę w domu, potrzebujemy kilku niezbędnych składników.... "
  }}
]
"""

human_template = "{text}"


def generate_blog_content(provided_text, log):
    log.info(f"provided_text:{provided_text}")
    try:
        chat = ChatOpenAI(model_name="gpt-3.5-turbo")
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("human", human_template),
            ]
        )
        formatted_chat_prompt = chat_prompt.format_messages(text=provided_text)
        log.info(f"prompt: {formatted_chat_prompt}")
        ai_response = chat.predict_messages(formatted_chat_prompt)
        log.info(f"content: {ai_response}")
        content_json = json.loads(ai_response.content)
        log.info(f"content_json: {content_json}")
        content_list = [item["content"] for item in content_json]
        return content_list
    except Exception as e:
        log.error(f"Exception: {e}")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    log = configure_logger("blogger_openai")
    text = (
        "'Wstęp: kilka słów na temat historii pizzy', 'Niezbędne składniki na pizzę', 'Robienie pizzy', 'Pieczenie "
        "pizzy w piekarniku'"
    )
    generate_blog_content(text, log)

import json

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from common.logger_setup import configure_logger

# Perform a task called liar. This is a mechanism that says off topic 1/3 of the time. Your task is to send your
# question in English (any question, e.g. "What is capital of Poland?") to the endpoint /task/ in a field named
# 'question' (POST method, as a regular form field, NOT JSON). The API system will answer that question (in the 'answer'
# field) or start telling you about something completely different, changing the subject.
#
# Your task is to write a filtering system (Guardrails) that will determine (YES/NO) whether the answer is on topic.
# Then return your verdict to the checking system as a single YES/NO word.
# If you retrieve the content of the task through the API without sending any additional parameters,
# you will get a set of prompts.
# How to know if the answer is 'on topic'?
# If your question # was about the capital of Poland, and in the answer you receive a list of monuments in Rome,
# the answer to send to the API is NO


system_template = """
Given the following sample input:
    question: Where is Rome? 
    bot_answer: Star Wars is a movie.

Please validate the bot's answer and provide a binary JSON response containing either "YES" or "NO" using the following format:
```json
{{
  "validation": "NO"
}}
```
"""

human_template = """question: {human_question} 
bot_answer: {bot_answer} """


def validate_bot_answer(human_question, bot_answer, log):
    log.info(f"question:{human_question}, answer: {bot_answer}")
    try:

        chat = ChatOpenAI(model_name="gpt-3.5-turbo")
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("human", human_template),
            ]
        )
        formatted_chat_prompt = chat_prompt.format_messages(human_question=human_question, bot_answer=bot_answer)

        log.info(f"prompt: {formatted_chat_prompt}")
        ai_response = chat.predict_messages(formatted_chat_prompt)
        log.info(f"content: {ai_response}")
        content_json = json.loads(ai_response.content)
        log.info(f"content_json: {content_json}")
        open_ai_answer = content_json["validation"]
        return open_ai_answer
    except Exception as e:
        log.error(f"Exception: {e}")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    log = configure_logger("blogger_openai")
    question = "Co jest stolicą Polski?"
    answer = "In Italy, where pizza originated, the concept of pineapple on pizza is largely frowned upon."
    validate_bot_answer(question, answer, log)

    question = "Co jest stolicą Polski?"
    answer = "Warszawa"
    validate_bot_answer(question, answer, log)

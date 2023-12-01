import os
from dotenv import load_dotenv, find_dotenv
from icecream import ic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from common.constants import AI_DEVS_SERVER
from common.files_read_and_download import get_file_name, download_file, read_file_contents
from common.logger_setup import configure_logger

# TIP: The context is provided between ``` characters. You might wonder why?
# If markdown content would be passed then ### is fragment of markdown (### is used to create a heading level 3).

system_template = """

Return answer for the question in POLISH language.
Maximum length for the answer should be 200 characters.
Use only provided context.

Sample answer:
user: kim z zawodu jest Ernest?
assistant: Jest fryzjerem.

context: ```{context_value}``` """

user_template = """{user_question} """


def give_me_answer_based_on_context(usr_template=None,
                                    usr_question=None,
                                    sys_template=None,
                                    context_val=None,
                                    log=None, ):
    log.info(f"usr_question:{usr_question}")
    try:

        chat = ChatOpenAI(model_name="gpt-3.5-turbo")
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_template),
                ("human", usr_template),
            ]
        )
        get_name_formatted_chat_prompt = chat_prompt.format_messages(context_value=context_val,
                                                                     user_question=usr_question)
        log.info(f"prompt: {get_name_formatted_chat_prompt}")
        ai_response = chat.predict_messages(get_name_formatted_chat_prompt)
        log.info(f"content: {ai_response}")
        response_content = ai_response.content
        log.info(f"response_content: {response_content}")

        return response_content
    except Exception as e:
        log.error(f"Exception: {e}")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    log = configure_logger("scraper")

    try:
        endpoint = f'{AI_DEVS_SERVER}/text_pasta_history.txt'
        question = "komu przypisuje się przepis na danie lagana?"
        file = get_file_name(endpoint)
        if not os.path.exists(get_file_name(endpoint)):
            download_file(endpoint)
        context = read_file_contents(file)
        response = give_me_answer_based_on_context(user_template, question, system_template, context, log)
        ic(response)

        endpoint = f'{AI_DEVS_SERVER}/text_pizza_history.txt'
        question = "w którym roku według legendy została wynaleziona pizza Margherita?"
        file = get_file_name(endpoint)
        if not os.path.exists(file):
            download_file(endpoint)
        context = read_file_contents(file)
        response = give_me_answer_based_on_context(user_template, question, system_template, context, log)
        ic(response)

    except Exception as e:
        log.exception(f'Exception: {e}')

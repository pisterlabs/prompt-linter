import json
import os
from dotenv import load_dotenv, find_dotenv
from icecream import ic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from common.files_read_and_download import get_file_name, download_file, read_file_contents
from common.logger_setup import configure_logger

# TIP: The context is provided between ``` characters. You might wonder why?
# If markdown content would be passed then ### is fragment of markdown (### is used to create a heading level 3).

system_template = """

Answer questions as truthfully as possible using the context below and nothing else 
If you don't know the answer, say: I don't know.

context: ```{context_value}``` """

user_template = """{user_question} """


def give_me_answer_based_on_context(usr_template=None,
                                    usr_question=None,
                                    sys_template=None,
                                    context_val=None,
                                    log=None):
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


def get_data_dictionary(url):
    downloaded_file = get_file_name(url)
    if not os.path.exists(downloaded_file):
        download_file(url)

    people = json.loads(read_file_contents(downloaded_file))

    result_dict = {}

    for person in people:
        name_surname = person['imie'] + ' ' + person['nazwisko']
        rest_of_data = {key: value for key, value in person.items() if key not in ['imie', 'nazwisko']}
        result_dict[name_surname.upper()] = rest_of_data

    return result_dict


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    log = configure_logger("people")
    url = 'https://zadania.aidevs.pl/data/people.json'
    people_dict = get_data_dictionary(url)
    context = ""

    try:
        question = 'Gdzie mieszka Zofia Bzik?'
        system_template = read_file_contents('name_category_system_prompt.txt')
        ai_response = give_me_answer_based_on_context(user_template, question, system_template, context, log)
        if "I don't know" in ai_response:
            raise ValueError
        name, category = ai_response.strip().split(';')
        name = name.strip().upper()
        category = category.strip()

        ic(name.upper())
        answer = people_dict[name].get(category)

        ic(answer)
        final_response = answer
        ic(final_response)


    except Exception as e:
        log.exception(f'Exception: {e}')

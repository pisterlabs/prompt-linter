import os
from dotenv import load_dotenv, find_dotenv
from icecream import ic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from common.constants import AI_DEVS_SERVER
from common.files_read_and_download import get_file_name, download_file, read_file_contents
from common.logger_setup import configure_logger
from tasks.c03l04.qdrant_embeddings_full import QdrantVectorStore

# TIP: The context is provided between ``` characters. You might wonder why?
# If markdown content would be passed then ### is fragment of markdown (### is used to create a heading level 3).

system_template = """

You need to guess link based on context. 
If you are not sure answer: I don't know.

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


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    log = configure_logger("scraper")

    context = ""
    qdrant_host = 'localhost'
    qdrant_port = 6333
    qdrant_collection_name = 'unknow_news'
    ai_vector_size = 1536

    vector_db_obj = QdrantVectorStore(host=qdrant_host,
                                      port=qdrant_port,
                                      collection_name=qdrant_collection_name,
                                      vector_size=ai_vector_size)

    try:
        question = 'Co różni pseudonimizację od anonimizowania danych?'
        hints = vector_db_obj.search_using_embedded_query(question)

        ic(hints)

        # if "I don't know" in response:
        #     raise ValueError
        # else:
        #     final_response = response
        #     ic(final_response)

    except Exception as e:
        log.exception(f'Exception: {e}')

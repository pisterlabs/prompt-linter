''' This chain is used to optimize the question by using the content of the question. '''
import os
import langchain
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.cache import SQLiteCache
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

SYS_PATH_LOCAL = '/workspaces/b3rn_zero_streamlit'
SYS_PATH_STREAMLIT = '/app/b3rn_zero_streamlit/'
SYS_PATH = SYS_PATH_STREAMLIT
langchain.llm_cache = SQLiteCache(database_path=f"{SYS_PATH}/data/langchain_cache.db")

system_message_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        template="""
# Your role and task
Rephrase the Human questions to align with the standards of a Swiss social insurance expert. 
The restructured question should elicit the same response as the original but with enhanced 
clarity and precision. Answer not the question, rephrase it. 
You response should be in german.

# Examples of good questions:
Wie hoch ist der aktuelle AHV-Rentenbetrag in der Schweiz?
Welche Voraussetzungen müssen erfüllt sein um eine IV-Rente zu erhalten?
Welche Leistungen werden durch die Erwerbsersatzordnung (EO) abgedeckt?

# Use Chunks
Use the "Chunks" content to refine the question.
Ensure you filter out irrelevant information and focus only on pertinent details.

## Chunks content:
    {chunks}
""",
        input_variables=["chunks"],
    )
)
human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="""
====================
Frage: {question}
====================
Generiere zwei sehr ähnliche mögliche Fragen. du kannst die Fragen mit einem Komma trennen.
die frage welche die ursprünglich Frage am besten präzisiert nennst du als erstes:
""",
        input_variables=["question"],
    )
)
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])


def optimize_question(user_input, openai_api_key, sys_path):
    ''' optimize the question by using the content of the question. '''

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    new_db1 = FAISS.load_local(f'{sys_path}/data/vectorstores/eak_admin_ch_defaultdocs_faiss_index_4096',
                               embeddings)
    new_db2 = FAISS.load_local(f'{sys_path}/data/vectorstores/eak_admin_ch_defaultdocs_faiss_index_512',
                               embeddings)
    new_db3 = FAISS.load_local(f'{sys_path}/data/vectorstores/ch_ch_texts_faiss_index_4096',
                               embeddings)

    new_db1.merge_from(new_db2)
    new_db1.merge_from(new_db3)
    new_db = new_db1

    chat = ChatOpenAI(
        temperature=0.8,
        model="gpt-4",
        openai_api_key=openai_api_key)

    chain = LLMChain(
        llm=chat,
        prompt=chat_prompt_template,
        verbose=False)

    docs = new_db.similarity_search(user_input, k=10)
    thechunk = ""
    for doc in docs:
        thechunk += doc.page_content + "\n-------end this content-----------\n\n"
    return chain.run(chunks=thechunk, question=user_input)


if __name__ == "__main__":
    QUESTION = "was isch ch.ch"
    optimized_question = optimize_question(QUESTION)
    print(optimized_question)

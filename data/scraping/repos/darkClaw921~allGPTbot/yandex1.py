import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from YaGPT import YandexGPTEmbeddings
from YaGPT import YandexLLM
from dotenv import load_dotenv
import os

load_dotenv()
token = os.environ.get('YC_IAM_TOKEN')

# token = 


instructions = """
Представь себе, что ты сотрудник Yandex Cloud. Твоя задача - вежливо и по мере своих сил отвечать на все вопросы собеседника."""

llmYandex = YandexLLM(iam_token = token,
                instruction_text = instructions,
                folder_id = "b1g76nm1veej4ag3kmmp")
# a = llmYandex._call("Привет, как дела?")
# print(a)
# 1/0


# # Промпт для обработки документов
# document_prompt = langchain.prompts.PromptTemplate(
#     input_variables=["page_content"], 
#     template="{page_content}"
# )

# # Промпт для языковой модели
# document_variable_name = "context"
# stuff_prompt_override = """
#     Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
#     Текст:
#     -----
#     {context}
#     -----
#     Вопрос:
#     {query}
# """
# prompt = langchain.prompts.PromptTemplate(
#     template=stuff_prompt_override,
#     input_variables=["context", "query"]
# )

# # Создаём цепочку
# llm_chain = langchain.chains.LLMChain(llm=llm, 
#                                       prompt=prompt)
# chain = langchain.chains.combine_documents.stuff.StuffDocumentsChain(
#     llm_chain=llm_chain,
#     document_prompt=document_prompt,
#     document_variable_name=document_variable_name,
# )
# chain.run(input_documents=docs, query=query)
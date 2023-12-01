
import os
from functools import reduce

from elasticsearch import Elasticsearch
from elasticsearch_database_chain import ElasticsearchDatabaseChain
from langchain.chat_models import ChatOpenAI

client = Elasticsearch(
    os.environ.get('ES_ENDPOINT'),
)

API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY, model_name='gpt-3.5-turbo-16k')
db_chain = ElasticsearchDatabaseChain.from_llm(llm=llm, db=client, verbose=False, sample_documents_in_index_info=10)


def ask(question):
    # result = db_chain({"question": question})
    result = db_chain.run(question)
    return result


def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() == 'exit':
            print('Exiting...')
            break
        else:
            try:
                print(ask(prompt))
            except Exception as e:
                print(e)


if __name__ == "__main__":
    get_prompt()

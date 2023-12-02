from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from typing import List

def load_data_from_url(url:str) -> List:
    """

    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


def main(url:str, open_api_key:str, temperature:int=0, model_name='gpt-3.5-turbo-16k'):
    """
    """
    docs = load_data_from_url(url=url)
    llm = ChatOpenAI(temperature=temperature, openai_api_key=open_api_key, model_name=model_name)
    chain = load_summarize_chain(llm, chain_type='stuff')
    summary = chain.run(docs)
    return summary


if __name__ == '__main__':
    url = ''
    openai_api_key = ''
    main(url=url, open_api_key=openai_api_key)
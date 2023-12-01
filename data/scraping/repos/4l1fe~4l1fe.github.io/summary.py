import traceback

import langchain
from langchain.llms import HuggingFaceHub 
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser, Document
from langchain.chains.summarize import load_summarize_chain
from diskcache import Cache

from constants import OPENAI_KEY_FILE, DISK_CACHE_DIR, HUGGINGFACE_KEY_FILE


cache = Cache(DISK_CACHE_DIR)
langchain.debug = True
langchain.verbose = True


class OutputParser(BaseOutputParser):

    def parse(self, text):
        return text


@cache.memoize()
def summarize(article_text):
    system_message = 'Summarize the following article text. '\
                      'Aticles have been written by a programmer about different technology subjects.'
    human_message = '{article_text}'

    chat_prompt = ChatPromptTemplate.from_messages(
                      [SystemMessagePromptTemplate.from_template(system_message),
                       HumanMessagePromptTemplate.from_template(human_message)])
    chain = LLMChain(
         llm=ChatOpenAI(openai_api_key=open(OPENAI_KEY_FILE, 'rt').read().strip(),
                        model_name='gpt-3.5-turbo-16k',
                        temperature=0),
         prompt=chat_prompt,
         output_parser=OutputParser()
     )

    try: 
        result = chain.run(article_text=article_text)
    except Exception:
        result = traceback.format_exc()
        
    return result


@cache.memoize()
def summarize_refine(text_chunks: list, model='huggingface'):
    docs = [Document(page_content=text_chunk) for text_chunk in text_chunks]

    # if model == 'huggingface':
    #     # llm = HuggingFaceHub(repo_id='facebook/bart-large-cnn',
    #     # llm = HuggingFaceHub(repo_id='meta-llama/Llama-2-7b',
    #     llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b',
    #                          model_kwargs={"temperature": 0, "max_length": 256},
    #                          huggingfacehub_api_token=open(HUGGINGFACE_KEY_FILE, 'rt').read().strip())
    # else:
    llm = ChatOpenAI(openai_api_key=open(OPENAI_KEY_FILE, 'rt').read().strip(),
                     model_name='gpt-3.5-turbo',
                     temperature=0)
    # chain = load_summarize_chain(llm, chain_type="refine")
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    try: 
        result = chain.run(docs)
    except Exception:
        result = traceback.format_exc()
        
    return result


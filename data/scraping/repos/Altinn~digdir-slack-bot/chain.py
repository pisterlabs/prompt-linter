'''
===========================================
        Module: Chain functions
===========================================
'''
import box
import yaml
import os

from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from code_qa.prompts import qa_template
from code_qa.llm import build_llm

# Import config vars
with open('code_qa/config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt

def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs={'prompt': prompt},      
                                       verbose=True                                 
                                       )
    return dbqa

def setup_dbqa():
    api_key = os.environ['OPENAI_API_KEY_ALTINN3_DEV']
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa

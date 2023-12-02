'''
===========================================
        Module: Chain functions
===========================================
'''
import box
import yaml

from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from .prompts import categorize_new_message
from .llm import build_llm
from .config_chain import config

cfg = config()

def set_choose_team_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=categorize_new_message,
                            input_variables=['question'])
    return prompt

def build_retrieval_choose_team(llm, prompt):
    dbqa = LLMChain(llm=llm, prompt=prompt, verbose=False)
    return dbqa


def setup_dbqa():
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    #                                    model_kwargs={'device': 'cpu'})
    # vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm()
    choose_team_prompt = set_choose_team_prompt()
    dbqa = build_retrieval_choose_team(llm, choose_team_prompt)

    return dbqa


def query(dbqa, user_input):
    if cfg.MODEL_TYPE == "gpt-4":
        return dbqa(user_input)
    else:
        return dbqa({'query': user_input})
    

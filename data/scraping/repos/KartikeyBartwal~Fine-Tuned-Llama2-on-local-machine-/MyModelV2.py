from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_CHROMA_PATH = "vectorstores/db_chroma"

custom_prompt_template = ''' Use the following pieces of information to answer the user's question. If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else
'''

def set_custom_prompt():
    '''
    prompt template for QA retrieval for each vector stores
    '''
    
    prompt = PromptTemplate(template = custom_prompt_template , 
                           input_variables = ["context" , "question"])
    
    return prompt

def load_llm():
    llm = CTransformers(
    
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm


def retrieval_qa_chain(llm , prompt , db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {'k' : 2}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt' : prompt}
    )
    
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2' , 
                                       model_kwargs = {'device' : 'cpu'})
    db = Chroma(persist_directory = DB_CHROMA_PATH , 
                 embedding_function = embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm , qa_prompt , db)
    
    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query' : query})
    return response


if __name__ == '__main__':
    qa_bot()
    
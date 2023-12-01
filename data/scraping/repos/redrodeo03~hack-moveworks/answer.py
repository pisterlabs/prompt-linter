import bardapi
import os
from langchain.document_loaders import TextLoader
from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import OpenLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
from decouple import config


def get_answer(input_text):

    os.environ["COHERE_API_KEY"] = config("COHERE_API_KEY")    
    token = config('TOKEN')

    embeddings = CohereEmbeddings()

    db = FAISS.load_local("cohere_index", embeddings)


    context = db.similarity_search(input_text)


    context_str = ""

    for i in context:
        print(i)
        context_str = context_str + i.page_content
    # prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    temp_prompt = f"You are a chatbot for the company Moveworks. Your job is to answer questions based on the context provided. Only use the following context to answer the question. If you do not know the answer of the the question say I don't know. Split the answer into segments. MAke sure to answer with proper punctuation and indentation like bold letters and bullets. \n Context: %s" %(context_str)
    question_prompt = f"\n Question: %s: " %(input_text)
    final_prompt = temp_prompt + question_prompt

    print(final_prompt)
    # Send an API request and get a response.
    answer = bardapi.core.Bard(token).get_answer(final_prompt)
    final_ans = answer['content']

    ans =''

    for i in final_ans:
        if (i != "*"):
            ans = ans + i

    return ans
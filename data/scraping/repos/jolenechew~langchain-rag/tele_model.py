# tried out tele bot

import os
import openai
import sys
import numpy as np
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from indexing import *

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_PROJECT"] = "Chatbot"

load_dotenv()

# get API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

def getResponse(question: str) -> str:

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 openai_api_key=OPENAI_API_KEY, temperature=0, request_timeout=120)

    # Define parameters for retrival
    retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k": 5})


    # Define template prompt
    template = """Use the following pieces of context to answer the question at the end. Always say Thanks for asking at the end of the answer.
    {context}
    Question: {question}"""

    your_prompt = PromptTemplate.from_template(template)

    # Execute chain
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        combine_docs_chain_kwargs={"prompt": your_prompt},
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        verbose=True,
        memory=memory
    )

    # Evaluate your chatbot with questions
    result = qa({"question": question})

    print(result)
    return result['answer']

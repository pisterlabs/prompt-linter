from langchain.document_loaders import DirectoryLoader, TextLoader
from transformers import AutoTokenizer
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import TFIDFRetriever
import numpy as np
import pickle
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import datetime
import os
import openai
import sys
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import start_config

class chatbot:
    def __init__(self):
        # Load the NumPy array from the file, allowing pickles
        with open("embeddings_10_15.npy", "rb") as f:
            embeddings_array = np.load(f, allow_pickle=True)
        
        # Convert the NumPy array back to a list of embeddings
        self.embeddings = embeddings_array.tolist()

        # Unpickle the documents
        with open("documents_10_15.pickle", "rb") as f:
            self.documents = pickle.load(f)

        self.tfidf_retriever = TFIDFRetriever.from_documents(documents=self.documents, embeddings=self.embeddings, k=10)

        key = start_config.chat_api_token

        sys.path.append('../..')
        _ = load_dotenv(find_dotenv())  # read local .env file
        os.environ['OPENAI_API_KEY'] = key
        openai.api_key = os.environ['OPENAI_API_KEY']

        current_date = datetime.datetime.now().date()
        if current_date < datetime.date(2023, 9, 2):
            llm_name = "gpt-3.5-turbo-0301"
        else:
            llm_name = "gpt-3.5-turbo"

        llm_name = "gpt-3.5-turbo-16k"

        self.llm = ChatOpenAI(model_name=llm_name, temperature=0)

        # Build prompt
        template = """Use the following pieces of context to answer the question at the end,and act like a legal advisor and always mention the article number in your answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer, always mention the article number in your answer.
        {context}
        Question: {question}
        Helpful Answer:"""
        # QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def answer_question(self, question):
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.tfidf_retriever
        )
        result = qa_chain({"query": question})
        return result["result"]


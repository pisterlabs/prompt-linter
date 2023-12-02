"""
Summarization Module
====================

This module provides a class for generating summaries of text using OpenAI's GPT language model.

"""

import os
import openai

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

from .SummarizationException import SummarizationException
from .summarizer import Summarizer


class Summarization(Summarizer):

    def __init__(self, llm=None):
        api_key = os.environ['OPENAI_API_KEY']

        if not api_key:
            raise ValueError("API key is not set. Please set OPENAI_API_KEY in your environment.")

        self.llm = llm or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", api_key=api_key)

    def get_text_chunks_into_docs(self, text: str) -> list[Document]:
        """
        Splits the given text into chunks using CharacterTextSplitter and returns a list of Document objects.

        **Parameters:**

        text (str): The text to be split into chunks.

        **Returns:**

        list[Document]: A list of Document objects representing the text chunks.

        """
        try:
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
            return docs
        except Exception as e:
            print(f"Error occured during text chunking: {e}")

    def get_summary(self, full_text: str) -> str:
        """
        Generates a summary using LLMChain for a given full text by breaking it into smaller chunks.

        **Parameters:**

        full_text (str): The full text to be summarized.

        **Returns:**

        str: The generated summary.

        """
        try:

            ## Split the full text into smaller text chunks
            text_chunks_into_docs = self.get_text_chunks_into_docs(full_text)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            all_splits = text_splitter.split_documents(text_chunks_into_docs)

            ## Create a vectorstore from the text chunks using Chroma and GPT-4 embeddings.
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

            ## Define a prompt for summarization.
            prompt = PromptTemplate.from_template(
                "Summarize in simple words. Include important details of the bill in 250 words: {docs}"
            )

            ## Create an LLMChain with the defined prompt and enable verbosity to debug.
            llm_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)

            ## Define a question to ask the model for summarization.
            question = "Write a summary within 200 words. "

            """ 
         Chroma vector store similarity search allows the LLM chain to focus on the relevant
         documents for summarization instead of passing all documents to the chain. 
         The retrieved documents are then used by the LLM chain to generate the summary using the prompt.
         """
            docs = vectorstore.similarity_search(question)

            ## Generate a summary using the LLMChain
            result = llm_chain(docs)

            ## Output
            return result["text"]
        except Exception as e:
            raise SummarizationException from e

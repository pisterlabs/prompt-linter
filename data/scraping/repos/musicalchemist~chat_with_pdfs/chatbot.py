from PyPDF2 import PdfReader
import spacy
import nltk
from annoy import AnnoyIndex
from transformers import AutoTokenizer, AutoModelForCausalLM
from io import BytesIO
import torch
import os
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

class PDFChatBot:
    def __init__(self):
        # load environment variables from .env file
        load_dotenv() #path to env file
        _ = load_dotenv(find_dotenv())
        self.OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
        # self.nlp = spacy.load("en_core_web_md")
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        # self.model = AutoModelForCausalLM.from_pretrained("TheBloke/vicuna-13B-1.1-HF")

    def read_pdf(self, pdf_files):
        """Read text data from multiple pdf files"""
        all_text = []
        for pdf_file in pdf_files:
            # Read binary content
            binary_content = pdf_file.read()
            # Create a BytesIO object and pass to PdfReader
            pdf_content = self.extract_pdf_content(BytesIO(binary_content))
            all_text.append(pdf_content)
        return " ".join(all_text)

    def extract_pdf_content(self, pdf_file):
        """Extract text data from a single pdf file"""
        pdf_content = []
        pdf = PdfReader(pdf_file)
        for page in pdf.pages:
            pdf_content.append(page.extract_text())
        return "\n".join(pdf_content)

    def partition_text(self, text, chunk_size=1500, chunk_overlap=300):
        """Split the text into chunks of a given size, with a given overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        return chunks

    def create_embedding_index(self, text_partitions):
        vectors = [self.nlp(partition).vector for partition in text_partitions]
        dimension = len(vectors[0])
        index = AnnoyIndex(dimension, 'angular')
        for i, vector in enumerate(vectors):
            index.add_item(i, vector)
        index.build(10)
        return index, text_partitions

    def create_embeddings(self, type="openai"):
        if type == "openai":
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = CohereEmbeddings()
        return embeddings

    def create_embedding_vectorstore(self, text_partitions, embeddings, type="faiss"):
        if type == "faiss":
            # vectorstore = FAISS.from_documents(text_partitions, embeddings)
            vectorstore = FAISS.from_texts(text_partitions, embeddings)
        else:
            # vectorstore = Chroma.from_documents(text_partitions, embeddings)
            vectorstore = Chroma.from_texts(text_partitions, embeddings)
        return vectorstore
    
    def get_docs(self, vectorstore, query):
        docs = vectorstore.similarity_search(query)
        return docs

    def generate_qa_chain(self, vectorstore):
        #instantiate prompt template
        template = """You are a chatbot having a conversation with a human. The human has uploaded pdf documents and will ask you questions about the information and content in those pdfs. Please use the uploaded pdfs, as well as any past conversation memory, as context for your reply.

        Here is the past chat history: 
        {chat_history}

        Human: {question}
        Chatbot:"""

        prompt = PromptTemplate(
                input_variables=["chat_history", 
                                 "question"], 
                                 template=template
            )

        memory = ConversationBufferMemory(
                memory_key="chat_history", 
                input_key='question', 
                output_key='answer', 
                return_messages=True
            )

        qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0, model="gpt-4"), 
                condense_question_prompt=prompt,
                retriever=vectorstore.as_retriever(), 
                chain_type='stuff',
                memory=memory, 
                return_source_documents=True
            )

        return qa_chain
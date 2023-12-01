from PyPDF2 import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from prompts import REFINE_PROMPT_QUESTIONS, PROMPT_QUESTIONS

# Function to load the data from the pdf

def load_data(uploaded_file):
    # Set up the pdf reader
    pdf_reader = PdfReader(uploaded_file)

    text =""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Create a function to split the text into chunks
def split_text(text, chunk_size, chunk_overlap):
    # Initialize text splitter
    text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=chunk_size, chunk_overlap= chunk_overlap)

    text_chunk = text_splitter.split_text(text)

    document = [Document(page_content=t) for t in text_chunk]

    return document

# Function initialize LLM
def initialize_llm(model, temperature):
    llm = ChatOpenAI(model=model, temperature=temperature)

    return llm

# Function to generate questions
def generate_questions(llm, chain_type, documents):
    # Initialize the question chain
    question_chain = load_summarize_chain(llm=llm, chain_type=chain_type, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS, verbose=True)

    questions = question_chain.run(documents)

    return questions

# Function to create retrieval question answer chain
def create_retrieval_qa_chain(documents, llm):
    embeddings = OpenAIEmbeddings()

    vector_database = Chroma.from_documents(documents=documents, embedding=embeddings)

    retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_database.as_retriever())

    return retrieval_qa_chain
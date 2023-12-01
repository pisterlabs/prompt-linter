from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFLoader
import PyPDF2
from langchain.memory import ConversationBufferMemory
from dotenv import dotenv_values
from langchain.chains import LLMChain  
from langchain.chains.question_answering import load_qa_chain  
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT 
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import os

config = dotenv_values(".env") 
llm = OpenAI(openai_api_key=config['OPENAI_API_KEYS'], temperature=0, max_tokens=1000)

text_splitter = CharacterTextSplitter()

os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEYS']
text_splitter = CharacterTextSplitter()

class EUElectionReport:
    def __init__(self):
        self.chain = None
        self.docsearch = None

report = {}

def initialize_docs():
    reader = PyPDF2.PdfReader("EU EOM NGA 2023 FR.pdf")
    number_of_pages = len(reader.pages)
    text = ''
    for page_num in range(number_of_pages):
        page = reader.pages[page_num]
        text += page.extract_text()

    # texts = text_splitter.split_text(text)
    text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts1 = text_splitter1.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=config['OPENAI_API_KEYS'])
    # docsearch = Chroma.from_texts(texts1, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts1))])
    # print(documents)
    # embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts1, embeddings)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), return_source_documents=True)
    return qa

# def initialize_docs1():
#     reader = PyPDF2.PdfReader("EU EOM NGA 2023 FR.pdf")
#     number_of_pages = len(reader.pages)
#     text = ''
#     for page_num in range(number_of_pages):
#         page = reader.pages[page_num]
#         text += page.extract_text()

#     # texts = text_splitter.split_text(text)
#     text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts1 = text_splitter1.split_text(text)
#     embeddings = OpenAIEmbeddings(openai_api_key=config['OPENAI_API_KEYS'])
#     vectorstore = Chroma.from_texts(texts1, embeddings)
#     llm = OpenAI(temperature=0)  
#     question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)  
#     doc_chain = load_qa_chain(llm, chain_type="map_reduce")  
    
#     chain = ConversationalRetrievalChain(  
#     retriever=vectorstore.as_retriever(),  
#     question_generator=question_generator,  
#     combine_docs_chain=doc_chain,  
#     )  
#     chat_history = []  
#     query = "who are the presidential canditates"  
#     result = chain({"question": query, "chat_history": chat_history})  

#     print(result['answer']) 

def summary_of_document():
    reader = PyPDF2.PdfReader("EU EOM NGA 2023 FR.pdf")
    number_of_pages = len(reader.pages)
    text = ''
    for page_num in range(number_of_pages):
        page = reader.pages[page_num]
        text += page.extract_text()

    # texts = text_splitter.split_text(text)
    text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts1 = text_splitter1.split_text(text)
    # print(texts1, 'print doc')
    prompt_template = """Write a Complete summary of the following with relevant examples:


    {text}

    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    docs = [Document(page_content=t) for t in texts1[:3]]
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)
    # print(docs, 'docs')
    result = chain.run({"input_documents": docs})
    # print(result, 'main result')
    return result

chain = initialize_docs()
def query_chain(question):
    # print(report)
    # summary_of_document()
    chat_history = []
    result = chain({'question': question, "chat_history": chat_history})
    # print(result, 'result')
    answer = result['answer']
    # source_documents = result['source_documents']
    # print("Answer:", answer)
    # print("Source Documents:", source_documents)
    return answer
    # print(initialize, 'initialized')
    # chat_history = []
    # if not report:
    #     initialize = initialize_docs()
    #     print(initialize, 'initialize')
    #     query_result = initialize.run({'question': question, "chat_history": chat_history})
    #     print(query_result)
    #     return query_result
    # print(report['chain'])
    # result = report['chain']({'question': question, "chat_history": chat_history})
    # print(result)
    # return result
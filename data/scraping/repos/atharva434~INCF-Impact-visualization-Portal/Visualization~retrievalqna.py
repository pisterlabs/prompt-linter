from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Cohere
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
import json
from pydantic import BaseModel, Field, validator

from langchain.output_parsers import PydanticOutputParser

llm = Cohere(temperature=0,cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi")
class Actor(BaseModel):
#     name: str = Field(description="name of an actor")
#     film_names: List[str] = Field(description="list of names of films they starred in")
    description: str = Field(description="detailed complete description of project")
    tech_stack: str = Field(description="Every single Required skills of the corresponding project")
    domain: str = Field(description="domain of the corresponding project")
    subdomain: str = Field(description="subdomain of the corresponding project")

parser = PydanticOutputParser(pydantic_object=Actor)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)



def ingest(urls):
    loader = UnstructuredURLLoader(urls=urls)
    # Split pages from pdf
    pages = loader.load()

    #  to store it in a folder name titan
    persist_directory = 'test'
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(pages)
    embeddings = CohereEmbeddings(cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi")
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    return True

# ingest(["https://en.wikipedia.org/wiki/Shah_Rukh_Khan","https://en.wikipedia.org/wiki/Jawan_(film)"])



def chat(query):
    


    persist_directory = 'test'

    # vectorstore = Chroma.from_documents(documents, embeddings)
    embeddings = CohereEmbeddings(cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    qa2 = ConversationalRetrievalChain.from_llm(Cohere(temperature=0,cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi"), vectorstore.as_retriever(),return_source_documents=True,
                                                memory=memory)
    result = qa2({"question": query})
    print(result["answer"])
    return result

def ingest_documents(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    # Split pages from pdf

    #  to store it in a folder name titan
    persist_directory = 'test_document'
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(pages)
    embeddings = CohereEmbeddings(cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi")
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    return True

def fill_db(org_name):
    persist_directory = 'test_document'
    embeddings = CohereEmbeddings(cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    # qa2 = ConversationalRetrievalChain.from_llm(Cohere(temperature=0,cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi"), vectorstore.as_retriever(),return_source_documents=True,
    #                                             memory=memory)
    retriever = vectorstore.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    # query=f"Give the complete description of {org_name}"
    
    # result = qa.run(query)
    query2= f"Give the description the Required skills, domain and subdomain of {org_name} in dictionary format"
    _input = prompt.format_prompt(query=query2)
    result2 = qa.run(_input.to_string())
    # print("hhi")
    # print(result)
    print(result2)
    result2=json.loads(result2)
    # out = parser.parse(result2)
    # print(out)
    # result2["description"]=result
    # params=
    # params["description"]=result
    #print(params)
    
    return result2


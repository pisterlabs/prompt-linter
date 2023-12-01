from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader,UnstructuredWordDocumentLoader,TextLoader,UnstructuredURLLoader
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,ConversationChain
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain import  SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
import os

MODEL = "gpt-3.5-turbo-0613"
K = 10

# def websurf():
#     urls = ['https://community.intersystems.com/']
    
#     loaders = UnstructuredURLLoader(urls=urls)
#     data = loaders.load()
#     print(data)


#connect to iris db 
def irisdb(query):
    
    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.

    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"

    The SQL query should NOT end with semi-colon
    Question: {input}"""

    PROMPT = PromptTemplate(
        input_variables=["input", "dialect"], template=_DEFAULT_TEMPLATE
    )

    db = SQLDatabase.from_uri("iris://superuser:SYS@localhost:1972/USER") 

    llm = OpenAI(temperature=0, verbose=True)

    db_chain = SQLDatabaseChain(llm=llm, database=db, prompt=PROMPT, verbose=True) 

    db_chain.run(query)

#Save document locally. by default use personal model name 
def ingest(path,apiKey,persist_directory = 'personal'):
    
    os.environ['OPENAI_API_KEY'] = apiKey
    embedding = OpenAIEmbeddings()

    fileType = getFileType(path)
    if fileType == "UNKOWN":
        return "Please provide PDF,DOC or TXT file to ingest"
    elif fileType == "PDF":
        loader = PyPDFLoader(path)       
    elif fileType == "DOC":
         ## Load and process the text
        loader = UnstructuredWordDocumentLoader(path)
    elif fileType == "TXT":
        loader = TextLoader(path)    
     
    try:
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        # Embed and store the texts
        # Supplying a persist_directory will store the embeddings on disk
        vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
        #save document locally
        vectordb.persist()
        vectordb = None
    except Exception as e:
        return e

    return "File uploaded successfully"

#function used in streamlit application
def docLoader(apiKey):
    ## Now we can load the persisted database from disk, and use it as normal. 
    os.environ['OPENAI_API_KEY'] = apiKey
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory='vectors', embedding_function=embedding)
    return vectordb

#function used in streamlit application
def contestLoader(apiKey):
    os.environ['OPENAI_API_KEY'] = apiKey
    embedding = OpenAIEmbeddings()
    ## Now we can load the persisted database from disk, and use it as normal. 
    vectordb = Chroma(persist_directory='contest', embedding_function=embedding)
    return vectordb
 
def irisdocs(query,apiKey):
    os.environ['OPENAI_API_KEY'] = apiKey
    embedding = OpenAIEmbeddings()
    persist_directory = 'vectors'
    ## Now we can load the persisted database from disk, and use it as normal. 
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectordb.as_retriever(), memory=memory)
    except Exception as e:
        return e
    return qa.run(query)

def iriscontest(query,apiKey):
    os.environ['OPENAI_API_KEY'] = apiKey
    embedding = OpenAIEmbeddings()
    persist_directory = 'contest'
    ## Now we can load the persisted database from disk, and use it as normal. 
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectordb.as_retriever(), memory=memory)
    except Exception as e:
        return e
    return qa.run(query)
#personal chatgpt
def irislocal(query,apiKey,persist_directory = 'personal'):    
    os.environ['OPENAI_API_KEY'] = apiKey
    embedding = OpenAIEmbeddings()
    ## Now we can load the persisted database from disk, and use it as normal. 
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectordb.as_retriever(), memory=memory)
    except Exception as e:
        return e
    return qa.run(query)

#use OpenAI 
def irisOpenAI(query,apiKey):
    os.environ['OPENAI_API_KEY'] = apiKey
    embedding = OpenAIEmbeddings()
    try:
        llm = ChatOpenAI(temperature=0,openai_api_key=apiKey, model_name=MODEL, verbose=False) 
        entity_memory = ConversationEntityMemory(llm=llm, k=K )
        qa = ConversationChain(llm=llm,   prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, memory=entity_memory)
    except Exception as e:  
        return e
    
    return qa.run(query)
  

#Get file type
def getFileType(filePath):
    filename, file_extension = os.path.splitext(filePath)
    if file_extension == '.pdf':
        return "PDF"
    elif file_extension == '.docx' or file_extension=='.doc':
        return "DOC"
    elif file_extension == '.txt':
        return "TXT"
    else:
        return "UNKOWN"

#save pdf and text files locally to be used to chatGPT locally 
def initdata():
     #print(ingest('data\RCOS.pdf','vectors'))
     #print(ingest('data\current_contest.docx','contest'))
    pass

print('Test')
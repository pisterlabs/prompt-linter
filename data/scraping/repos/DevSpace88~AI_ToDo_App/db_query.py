import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

CHROMA_DB_DIRECTORY = "chroma_db/ask_django_docs"


def check_file_content():
    filename = 'code_contents.txt'

    if os.path.exists(filename):
        with open(filename, 'r') as file:
            content = file.read()
        return content
    else:
        return False

def answer_query(query, context):
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        collection_name="ask_django_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIRECTORY
    )

    chat = ChatOpenAI(temperature=0)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"verbose": True}
    )
    
    if context and check_file_content():
        file = check_file_content()
        query = f"{context} \n This is my Code so far: \n {file} \n {query}"
    elif context:
        query = f"{context} {query}"  
    
    
    result = chain({"question": query}, return_only_outputs=True)
    return result
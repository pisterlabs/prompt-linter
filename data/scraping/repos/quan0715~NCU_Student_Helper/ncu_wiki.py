from pydantic import BaseModel, Field
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import chromadb

persistent_client = chromadb.PersistentClient()

# load_dotenv()
api_key = os.getenv('OPEN_AI_API_KEY')

PDF_PATH_NAME = "docs/food.pdf"


class Document(BaseModel):
    """Interface for interacting with a document."""
    page_content: str
    metadata: dict = Field(default_factory=dict)


def loadDocument(path: str):
    loader = PyPDFLoader(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    document = loader.load_and_split(splitter)
    vectordb = Chroma.from_documents(
        document,
        embedding=OpenAIEmbeddings(openai_api_key=api_key),
        persist_directory='./data'
    )
    vectordb.persist()
    # return vectordb


def load_all_documents():
    documents = []
    for file in os.listdir('line_bot_callback/docs'):
        if file.endswith('.pdf'):
            pdf_path = 'line_bot_callback/docs/' + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        # elif file.endswith('.docx') or file.endswith('.doc'):
        #     doc_path = './docs/' + file
        #     loader = Docx2txtLoader(doc_path)
        #     documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = 'line_bot_callback/docs/' + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())
        elif file.endswith('.csv'):
            text_path = 'line_bot_callback/docs/' + file
            loader = CSVLoader(text_path)
            documents.extend(loader.load())
        elif file.endswith('.json'):
            json_path = 'line_bot_callback/docs/' + file
            loader = JSONLoader(json_path, ".[]", text_content=False)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        chunked_documents,
        embedding=OpenAIEmbeddings(openai_api_key=api_key),
        persist_directory='line_bot_callback/data'
    )
    vectordb.persist()
    return vectordb


def getWikiChainLLM():
    # db = loadDocument("docs/network_issue.pdf")
    # client = chromadb.PersistentClient(path="/path/to/save/to")
    db = load_all_documents()
    db = Chroma(persist_directory="line_bot_callback/data",
                embedding_function=OpenAIEmbeddings(openai_api_key=api_key))
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=api_key),
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
    )

    # result = chain({'query': '條列幾間中壢好吃的美食'})
    # print(result['result'])
    # result = chain({'query': '要如何知道自己的mac address呢'})
    # print(result['result'])
    # result = chain({
    #     'query': '給我火舞社相關資訊'
    # })

    # print(result['result'])
    # response = chain.run(input_documents=documents, question=query)

# if __name__ == "__main__":
#     runLLM()

    # embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, openai_api_key=api_key), vectorstore.as_retriever())

    # result = qa({"question": query, 'chat_history': []})
    # print('A:', result['answer'])

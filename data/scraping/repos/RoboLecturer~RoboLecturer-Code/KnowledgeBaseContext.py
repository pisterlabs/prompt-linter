import langchain
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import tiktoken
import pinecone
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key, pinecone_API_key


def KnowledgeBaseToPinecone(url,topic):
    """
    Upload PDF urls to pinecone index, for uploading large documents such as textbooks
    """

    index_name = 'pepper-memory'
    OPENAI_API_KEY = openai_API_key
    PINECONE_API_KEY = pinecone_API_key
    PINECONE_API_ENV = "us-central1-gcp"
    pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
    )

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


    loader = OnlinePDFLoader(url)
    data = loader.load()

    tokenizer = tiktoken.get_encoding('p50k_base')

    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(data)
    metadata_list = [{'topic': topic,'url' : url} for _ in range(len(chunks))]
    Pinecone.from_texts([t.page_content for t in chunks], embeddings, metadatas=metadata_list, index_name=index_name, namespace='script')










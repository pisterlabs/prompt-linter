import os
import textwrap
import pinecone
from langchain.text_splitter import SpacyTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Pinecone
"""
    LangChain is an open-source framework created to aid the development of applications 
    leveraging the power of large language models (LLMs). It can be used for chatbots, 
    text summarisation, data generation, code understanding, question answering, evaluation, 
    and more. Pinecone, on the other hand, is a fully managed vector database, making it easy 
    to build high-performance vector search applications without infrastructure hassles. 
    Once you have generated the vector embeddings using a service like OpenAI Embeddings, 
    you can store, manage and search through them in Pinecone to power semantic search, 
    recommendations, and other information retrieval use cases. See this post on LangChain 
    Embeddings for a primer on embeddings and sample use cases.
"""
"""
    Embeddings databases are just our raw pattern matching based on vectors. The reason we
    need these is to provide additional context and tokens that our model may not have seen.
    We also want to be able to search for similar documents based on the embeddings we have
    giving us additional efficiency and accuracy in our search.

    Out LLM is what generates the response as we can only search for documents.  Our query
    finds the document we indexeed with the highest probability of being the most similar
    and gives this back to us.
"""

os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "ad4be10d-f95c-4892-a38a-d5a107c756ca"
os.environ["PINECONE_ENV"] = "us-east4-gcp"

index_name = "biblegpt"
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def print_response(response: str):
    print("\n".join(textwrap.wrap(response, width=100)))

def load_embeddings(path):
    """
        Generate embeddings for the pages, insert into Pinecone vector database, 
        and expose the index in a retriever interface
        
        we want to create embeddings index to inform our model of our raw data it may not have 
        seen feeding our embeddings to our model will allow us to search for similar documents
        after creating it's own calculation of relations between documents
    """

    # index = VectorstoreIndexCreator().from_loaders([loader])
    loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)

    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536  
        )
    vectordb = Pinecone.from_documents(documents, embeddings, index_name=index_name)
    return vectordb

def search(searchQuery: str, docsearch):
    query = searchQuery
    docs = docsearch.similarity_search(query)
    print(docs[0].page_content)

if __name__ == "__main__":

    user_input = input("Do you want to process embeddings? (y/n): ")

    prompt = """
        Who was Noah?
    """

    if user_input.lower() == 'y':
        print("Continuing...")
        docsearch = load_embeddings('./library')
        search(prompt, docsearch)
    elif user_input.lower() == 'n':
        print("Skipping to search...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        docSearch = Pinecone.from_existing_index(index_name, embeddings)
        search(prompt, docSearch)
    else:
        print("Invalid input. Exiting...")
    

from langchain.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from requests.exceptions import ContentDecodingError

# API key
load_dotenv()

if __name__ == "__main__":
    

    # Define URLs to scrape
    urls = [
        'https://en.wikipedia.org/wiki/Tucson,_Arizona',
        'https://eu.azcentral.com/story/news/local/arizona/2023/08/26/over-120k-homes-at-risk-of-wildfire-damage-in-arizona/70672259007/',
        'https://wildlifeinformer.com/wild-animals-in-arizona/',
        'https://www.azcentral.com/story/news/local/arizona-wildfires/2023/06/09/arizona-wildfires-2023-map-where-fires-are-burning-now/70307114007/'
    ]

    # Load data from URLs
    loader = WebBaseLoader(urls)

    try:
        data = loader.load()
    except ContentDecodingError as e:
        raise e

    # Split data into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    # Generate embeddings for each chunk
    embeddings = OpenAIEmbeddings()

    # Create a vector store from the embeddings
    db = FAISS.from_documents(docs, embeddings)

    db.save_local("arizona")
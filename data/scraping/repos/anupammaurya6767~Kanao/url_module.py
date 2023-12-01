from bs4 import BeautifulSoup
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from kanao.modules.get_api_key import get_api_key

class TextDocument:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}  # Add an empty metadata attribute

def process_url(url):
    api_key = get_api_key()
    
    if not api_key:
        raise ValueError('OpenAI API key is not provided in the configuration file.')
    
    # Fetch the HTML content of the URL.
    response = requests.get(url)
    html = response.content

    # Parse the HTML content.
    soup = BeautifulSoup(html, "html.parser")

    # Extract the text from the HTML content.
    text = soup.get_text()

    # Create a TextDocument object with the extracted text
    document = TextDocument(page_content=text)

    # Initialize OpenAIEmbeddings for text embeddings
    embeddings = OpenAIEmbeddings()

    # Create a list with one TextDocument object
    documents = [document]

    # Create a ConversationalRetrievalChain with ChatOpenAI language model
    # and plain text search retriever
    txt_search = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3),
        retriever=txt_search.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
    )

    return chain

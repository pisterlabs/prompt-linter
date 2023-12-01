import os
import logging
import requests
from rq import Retry
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings

from andes.models import WebPage, WebPageChatHistory
from andes.utils.config import UPLOAD_DIRECTORY
from andes.services.serialization import pickle_dump, pickle_load
from andes.services.rq import QUEUES
from andes.prompts import FIN_QA_PROMPT


QA_PROMPT = PromptTemplate(
    template=FIN_QA_PROMPT, 
    input_variables=["context", 'question'],
    template_format='jinja2'
)


def create_webpage(url: str):
    """
    create WebPage sqlalchemy model object
    """
    page = WebPage(url=url)
    page.save()
    return page


def get_webpage(id: str):
    """
    get WebPage sqlalchemy model object
    """
    page = WebPage.query.get(id)
    return page


def crawl(page: WebPage):
    """
    crawl webpage and save HTML to a file
    """
    logging.info(f"Crawling WebPage {page}")

    # crawl webpage using python libraries
    response = requests.get(page.url)

    # get webpage title
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # extract the title
    title = soup.title.string

    # add title to page and save
    page.title = title
    page.save()
    
    # save raw HTML to a file
    filepath = os.path.join(UPLOAD_DIRECTORY, page.id, 'raw.html')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(response.content)

    # enqueue the create index task
    QUEUES['webpage_index_gen'].enqueue(create_index, page, retry=Retry(max=3))


def enqueue_crawl(page: WebPage):
    # enqueue the crawl task
    QUEUES['crawler'].enqueue(crawl, page, retry=Retry(max=3))
    logging.info(f"Enqueued crawl task for {page}")


def _split_webpage(filepath: str, chunk_size=4000, chunk_overlap=50) -> list[str]:
    """
    read raw HTML from file and split into chunks
    """
    with open(filepath, 'rb') as f:
        raw_webpage_html = f.read()

    # read raw HTML into string
    soup = BeautifulSoup(raw_webpage_html, 'html.parser')
    page_text = soup.get_text()

    # split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
        add_start_index = True,
    )

    splits = text_splitter.create_documents([page_text])
    texts = [split.page_content for split in splits]
    return texts


def create_index(page: WebPage):
    """
    create a langchain index for the webpage
    """
    logging.info(f"Started creating index for {page}")

    filepath = os.path.join(UPLOAD_DIRECTORY, page.id, 'raw.html')
    page_splits = _split_webpage(filepath)

    # create a langchain index for each chunk
    logging.info(f"Building index for {page}")
    embeddings = OpenAIEmbeddings()
    index = FAISS.from_texts(page_splits, embeddings)

    # save the index to disk
    index_path = os.path.join(UPLOAD_DIRECTORY, page.id, 'index.pkl')
    pickle_dump(index, index_path)


def enqueue_index_gen(page: WebPage):
    """
    enqueue the index generation task into a redis queue
    """
    QUEUES['index_gen'].enqueue(create_index, page, retry=Retry(max=3))
    logging.info(f"Enqueued index generation for {page.filename}")


def chat(page: WebPage, message: str) -> str:
    # query openai on the langchain index

    # sanity checks
    if not os.path.exists(os.path.join(UPLOAD_DIRECTORY, page.id, 'index.pkl')):
        raise ValueError("Index does not exist for this webpage")
    
    assert message is not None, "Message cannot be empty"

    # load the index from disk
    index_path = os.path.join(UPLOAD_DIRECTORY, page.id, 'index.pkl')
    index = pickle_load(index_path)

    # create a chain
    qa = RetrievalQA.from_chain_type(
        OpenAI(temperature=0), 
        chain_type="stuff", 
        retriever=index.as_retriever(),
        chain_type_kwargs={
            "prompt": QA_PROMPT
        }
    )

    # query GPT
    response = qa.run(message)

    # save the chat history
    WebPageChatHistory(
        webpage_id = page.id,
        question = message,
        answer = response
    ).save()

    return response

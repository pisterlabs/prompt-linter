import os
import logging
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFium2Loader
from bs4 import BeautifulSoup
import nltk

'''
This script is used to rebuild the embeddings database from the scraped documents.
'''

# initialize standard logger in debug mode
logging.basicConfig(level=logging.INFO)


def get_file_list(directory_path) -> list:
    # initialize empty list
    file_list = []

    with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_file():
                filename = entry.name
                file_list.append(filename)

    print(f'total walked: {len(file_list)}')
    return file_list


def extract_text(soup: BeautifulSoup) -> str:
    logger = logging.getLogger()

    logger.debug("Extract text")
    # List of tags to be removed directly
    for tag in soup(['script', 'style', 'meta', 'link', 'noscript', 'cdata']):
        tag.decompose()

    # Class names that indicate elements to be removed
    classes_to_remove = ['footer', 'header', 'nav', 'aside', 'sidebar', 'menu']
    for class_name in classes_to_remove:
        for div in soup.find_all("div", class_=class_name):
            div.decompose()

    # Get the text from the page
    items = [item.text for item in soup.select('p, ol li')]
    # turn items into string
    text = ' '.join(items)

    # Removing extra spaces and joining text
    cleaned_text = '\n'.join(line.strip() for line in text.strip().splitlines() if line.strip())
    return cleaned_text


def is_dense(document, min_tokens=300):
    content = document.page_content
    tokens = nltk.word_tokenize(content)
    return len(tokens) > min_tokens



def split_embed_store(splitter: RecursiveCharacterTextSplitter, db: Chroma, filename: str, ext: str) -> None:
    logger = logging.getLogger()
    logger.info("Split, embed and store " + filename)

    loader = None
    try:
        if ext == '.pdf':
            # loader = PyPDFium2Loader(filename)
            loader = PyPDFium2Loader(file_path=filename, extract_images=False)
            logger.debug('Load PDF')

        else:
            loader = TextLoader(filename)
            logger.debug('Load Text')
        if loader is not None:
            # docs = loader.load_and_split(splitter)

            # load the documents without split
            docs = loader.load()

            # remove the documents that are not dense
            docs = [doc for doc in docs if is_dense(doc)]

            if (len(docs)) == 0:
                logger.info(f'No dense documents in {filename}')
                return

            split_documents = []
            # remove from each document lines that are too short
            for doc in docs:
                lines = doc.page_content.split('\n')
                lines = [line for line in lines if len(line) > 50]
                doc.page_content = '\n'.join(lines)

            # split the documents
            splitdocs = splitter.split_documents(docs)
            split_documents.append(splitdocs)

            # embed and store
            db.add_documents(documents=splitdocs)
    except Exception as e:
        logger.error(f'Failed to split and embed: {filename}')
        logger.error(f'Exception: {e}')


# this is the main entry point for the program
if __name__ == "__main__":

    # initialized scraped directory variable
    scraped_directory = './scraped/'

    # get the list of files from get_file_list
    files_to_process = get_file_list(scraped_directory)

    # use openai embeddings
    embeddings = OpenAIEmbeddings()

    # create chroma db
    chroma_db = Chroma(persist_directory="./chroma_clean_ada", embedding_function=embeddings)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    # files_to_process = {'city-of-orlando-fy2003-2004-budget-book.pdf'}
    # files_to_process = {'www_orlando_gov-Parks-the-Environment-Directory-Songbird-Park.txt'}

    # iterate over the files
    for filename in files_to_process:
        # get the extension
        ext = os.path.splitext(filename)[1]
        # split, embed and store
        split_embed_store(text_splitter, chroma_db, scraped_directory + filename, ext)

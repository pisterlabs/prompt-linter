from langchain.document_loaders import UnstructuredURLLoader
from config import setup_logging

# Set up logging
logger = setup_logging()

def load_url(urls):
    logger.info(f'Loading data for {urls}')
    urls_loader = UnstructuredURLLoader(urls=urls)
    urls_data = urls_loader.load()
    return urls_data

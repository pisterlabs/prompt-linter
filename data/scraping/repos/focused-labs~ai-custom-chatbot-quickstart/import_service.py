from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from llama_index import VectorStoreIndex, download_loader, LLMPredictor, ServiceContext

from config import CHAT_MODEL
from pinecone_database import get_pinecone_storage_context

load_dotenv()


def get_llm_predictor():
    return LLMPredictor(llm=ChatOpenAI(temperature=0, max_tokens=512, model_name=CHAT_MODEL))


def get_service_context():
    llm_predictor_chatgpt = get_llm_predictor()
    return ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt)


def import_web_scrape_data(urls: list):
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=urls)

    index = VectorStoreIndex.from_documents(documents,
                                            storage_context=get_pinecone_storage_context(),
                                            service_context=get_service_context())
    return index

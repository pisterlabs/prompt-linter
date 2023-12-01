from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from loguru import logger
import yaml


def from_yaml(filepath):
    with open(filepath, "r", encoding="utf8") as ymlfile:
        try:
            data = yaml.safe_load(ymlfile)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading config file: {exc}")
            raise exc
        kwargs = {
            "data_path": "." + data.get("DATA_PATH"),
            "chunk_size": data.get("CHUNK_SIZE"),
            "chunk_overlap": data.get("CHUNK_OVERLAP"),
            "chat_embed_model": data.get("CHAT_EMBED_MODEL"),
            "db_path": "." + data.get("DB_PATH"),
        }
        return kwargs


def run_db_build():
    # Load configuration from YAML file
    logger.info("Loading configuration")
    config = from_yaml("./ai_driver/config/config.yaml")
    logger.info(config)
    # Build vector database
    logger.info("Building vector database")
    loader = DirectoryLoader(config["data_path"], glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"]
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=config["chat_embed_model"],
        model_kwargs={"device": "cpu"},
    )

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(config["db_path"])
    logger.info("Vectorestore Init Done!")


if __name__ == "__main__":
    run_db_build()

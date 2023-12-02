import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from libs.using_ffm import get_embed
import loguru

log = loguru.logger

GENERATED_JSON_FILE_PATH = "/home/ubuntu/AFS_tools/use_cases/webcrawler_chatbot/generated/website.json"
log.info(f"Doing embedding for {GENERATED_JSON_FILE_PATH}")

embeddings_zh = get_embed()

splitFunc = RecursiveCharacterTextSplitter(separators='',
                                           chunk_size=750,
                                           chunk_overlap=0,
                                           length_function=len)

start_time = time.time()


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["url"] = record["url"]
    metadata["title"] = record["title"]
    return metadata


loader = JSONLoader(
    file_path=GENERATED_JSON_FILE_PATH,
    jq_schema='.[]',
    content_key="body",
    metadata_func=metadata_func,)

datasets = loader.load_and_split(text_splitter=splitFunc)

doc_embedding = FAISS.from_documents(
    documents=datasets, embedding=embeddings_zh)

doc_embedding.save_local(f"/home/ubuntu/AFS_tools/use_cases/webcrawler_chatbot/generated/all_docs_embedding_website")

end_time = time.time()
running_time = end_time - start_time
print(f"Running time: {running_time} seconds")
print(f"FAISS vector store saved.")

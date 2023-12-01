import hashlib
import os
import json
import gzip
import logging

import deeplake

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.cache import SQLiteCache
from langchain.vectorstores import DeepLake
from langchain.globals import set_llm_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

activeloop_token = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTY5ODQ3MDIyMCwiZXhwIjoxNzMwMDkyNjEyfQ.eyJpZCI6ImFuYW50In0.AdMpABWtskDwWx4joLZhY9b9sqjNXvudeG2FqirATWFyRRczJJluCxnIt8udI3Bjy2Au672dEmDsmkOK5V0ZLg'
username = 'anant'
os.environ["ACTIVELOOP_TOKEN"] = activeloop_token

from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))


def _md5_hash_string(s):
    md5_hash = hashlib.md5()
    md5_hash.update(s.encode('utf-8'))
    return md5_hash.hexdigest()


def load_pdf_folder(path, images=False):
    return PyPDFDirectoryLoader(path, extract_images=images, silent_errors=True).load()


def dedup_documents(docs1, docs2):
    docs = docs1 + docs2
    docs_map = {_md5_hash_string(doc.page_content): doc for doc in docs}
    return docs_map.values()


def _compress_json(json_data):
    return gzip.compress(json_data.encode())


def compress_write(docs, path):
    jdocs = [json.dumps(doc.__dict__) for doc in docs]
    compressed = _compress_json(json.dumps(jdocs))
    with gzip.open(path, "wb") as f:
        f.write(compressed)


def get_deeplake(path, embeddings, overwrite=False, readonly=False):
    db = DeepLake(
        dataset_path=path, embedding=embeddings, overwrite=overwrite, readonly=readonly
    )
    return db


def copy_deeplake_hub(local_path, hub_path, token):
    al_db = deeplake.load(local_path)
    deeplake.copy(al_db, hub_path, overwrite=True)
    al_db.summary()


def main():
    logging.info("Starting main function")
    pdfs_folder = "flying_pdfs/"
    pdf_no_img = load_pdf_folder(pdfs_folder, images=False)
    logging.info("Loaded pdfs without images")
    pdf_img = load_pdf_folder(pdfs_folder, images=True)
    logging.info("Loaded pdfs with images")
    pdfs = dedup_documents(pdf_no_img, pdf_img)
    logging.info("Deduped pdfs")
    compress_write(pdf, "pdf.json.gz")
    logging.info("Wrote compressed pdfs")
    embeddings = OpenAIEmbeddings(show_progress_bar=True, model_kwargs={'batch_size': 50})
    db = get_deeplake("./deeplake", embeddings, overwrite=False, readonly=False)
    logging.info("Got deeplake")
    db.add_documents(pdfs)
    logging.info("Added documents to deeplake")
    copy_deeplake_hub("./deeplake", "hub://flying/test1", activeloop_token)
    logging.info("Finished main function")


if __name__ == "__main__":
    main()

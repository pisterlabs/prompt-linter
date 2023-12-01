from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path
import logging
import openai
import time
from . import config
logger = logging.getLogger('heymans')


class SigmundVectorStore(FAISS):
    
    def similarity_search(self, query, k=4, filter=None, fetch_k=20, **kwargs):
        if hasattr(config, 'add_context'):
            docs = config.add_context(query)
            k -= len(docs)
        else:
            docs = []
        docs += super().similarity_search(query, k, filter, fetch_k, **kwargs)
        return docs


def load_library(force_reindex=False):
    db_cache = Path('.db.cache')
    src_path = Path('sources')
    embeddings_model = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
    if not force_reindex and db_cache.exists():
        logger.info('loading library from cache')
        db = SigmundVectorStore.load_local(db_cache, embeddings_model)
    else:
        from langchain.document_loaders import TextLoader, PyPDFLoader, \
            JSONLoader
        logger.info('initializing library')
        data = []
        # Course are organized a text files per chapter and section. This is so
        # that they can also be used for practice
        for course in config.course_content.keys():
            sections = list((src_path / course).glob('**/*.txt'))
            logger.info(f'indexing course: {course}: {len(sections)} sections')
            data += [TextLoader(section).load()[0] for section in sections]
        # PDF files are unstructured. They can be named through config.sources
        for src in src_path.glob('pdf/**/*.pdf'):
            logger.info(f'indexing pdf: {src}')
            data += PyPDFLoader(str(src)).load_and_split()
        # jsonl is mainly for documentation
        for src in src_path.glob('jsonl/*.jsonl'):
            logger.info(f'indexing json: {src}')
            loader = JSONLoader(src, jq_schema='', content_key='content',
                                json_lines=True,
                                metadata_func=_extract_metadata)
            data += loader.load()
        # To avoid running into rate limits, we throttle the ingestion of the
        # documents
        for i in range(0, len(data), config.chunk_size):
            logger.info(
                f'ingesting chunk {i}-{i + config.chunk_size}/{len(data)}')
            chunk = data[i:i + config.chunk_size]
            if not i:
                db = SigmundVectorStore.from_documents(chunk, embeddings_model)
            else:
                time.sleep(config.chunk_throttle)
                db.add_documents(chunk)
        db.save_local(db_cache)
        logger.info(f'libary initialized')
    return db.as_retriever()


def _extract_metadata(record, metadata):
    metadata['url'] = record['url']
    metadata['title'] = record['title']
    return metadata

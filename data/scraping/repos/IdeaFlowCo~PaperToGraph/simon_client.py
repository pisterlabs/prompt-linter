import asyncio
import json
import logging
import time

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from psycopg2 import connect

from simon import AgentContext, Search
from simon.components.documents import parse_text, index_document

import gdrive
import utils
from utils import log_msg


class SimonClient:
    def __init__(self, config, uid_override=None):
        self.config = config
        self._context = self._agent_context_from_config(config, uid_override=uid_override)
        self._search_client = Search(self._context)

    def _agent_context_from_config(self, config, uid_override=None):
        openai_api_key = config['OPENAI_API_KEY']
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)
        reason_llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0)
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")

        pg = connect(**config['postgres'])

        if uid_override:
            uid = uid_override
        else:
            uid = config['SIMON_UID']

        context = AgentContext(llm, reason_llm, embedding, pg, uid)
        return context

    async def query_simon(self, query):
        logging.info(f'Querying Simon with query: "{query}"')
        a = time.time()
        result = await asyncio.to_thread(lambda: self._search_client.query(query))
        b = time.time()
        res_for_logs = json.dumps(result, indent=2)
        logging.info(f'Simon query completed in {(a-b):.2f} seconds. Result:\n{res_for_logs}')

        return result

    async def ingest_gdrive_file(self, gdrive_creds, file_id, max_retries=3):
        file = await gdrive.aget_file(credentials=gdrive_creds, file_id=file_id)
        if file is None:
            logging.error(f'Failed to get file with id {file_id}, skipping ingestion')
            return
        file_name = file['metadata']['name']
        logging.info(f'Parsing file {file_name} ({file_id})...')
        retries = 0
        while retries < max_retries:
            try:
                parsed_doc = await asyncio.to_thread(parse_text, file['content'], title=file_name, source=f'gdrive:{file_id}')
                break
            except Exception as err:
                retries += 1
                logging.error(f'Error while parsing file {file_name} ({file_id}): {err}')
                continue

        if retries == max_retries:
            logging.error(
                f'Failed to parse file {file_name} ({file_id}) after {max_retries} retries, skipping ingestion')
            return

        logging.info(f'Indexing file {file_name} ({file_id})...')

        retries = 0
        while retries < max_retries:
            try:
                await asyncio.to_thread(index_document, parsed_doc, context=self._context)
                break
            except Exception as err:
                retries += 1
                logging.error(f'Error while indexing file {file_name} ({file_id}): {err}')
                continue

        if retries == max_retries:
            logging.error(
                f'Failed to index file {file_name} ({file_id}) after {max_retries} retries, skipping ingestion')
            return

    async def ingest_gdrive_file_set(self, gdrive_creds, files):
        for f in files:
            await self.ingest_gdrive_file(gdrive_creds, f)

        return {'status': 'success'}


if __name__ == '__main__':
    config = utils.environment.load_config()
    utils.setup_logger(**config['logger'])
    log_msg('Logger initialized')

    client = SimonClient(config)

    start_time = time.time()
    result = asyncio.run(client.query_simon("What do I know about vestibular migraines?"))
    end_time = time.time()
    print(f'Simon query completed in {end_time - start_time:.2f} seconds')
    print(f'Query result:\n{json.dumps(result, indent=2)}')

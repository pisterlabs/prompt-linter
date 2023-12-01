import pinecone
from tqdm.auto import tqdm
from uuid import uuid4
from decouple import config
from langchain.embeddings.openai import OpenAIEmbeddings
from constants import EMBEDDING_MODEL
from langchain.document_loaders import TextLoader
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging, coloredlogs
load_dotenv(find_dotenv())

# ------------constants
BATCH_SIZE = 100

# --------------setup
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', default='INFO'))

logger.info('Setup')
embed = OpenAIEmbeddings(model=EMBEDDING_MODEL)

pinecone.init(
    api_key=config('PINECONE_API_KEY'),  # find api key in console at app.pinecone.io
    environment=config('PINECONE_ENV')  # find next to api key in console
)

# # delete index if it exists
# pinecone.delete_index(config('PINECONE_INDEX_NAME'))
# create a new index
# pinecone.create_index(
#     name=config('PINECONE_INDEX_NAME'),
#     metric='dotproduct', # dotproduct bc the embeddings are normalized = 1 (see here: https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use)
#     dimension=1536 # 1536 dim of text-embedding-ada-002
# )

index = pinecone.Index(config('PINECONE_INDEX_NAME'))


def create_index(index, folder_path):
    logger.info(f'index stats before we start: \b{index.describe_index_stats()}')

    txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]
    for filename in tqdm(txt_files):
        logger.info(f'Loading and Splitting Book: {filename}')
        file_path = os.path.join(folder_path, filename)
        loader = TextLoader(file_path=file_path, autodetect_encoding=True)
        book = loader.load()

        logger.debug('Splitting Book into Docs...')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(book)
        n_docs = len(docs)
        metadata = {
            'book': filename
        }

        logger.info('Running Batches to Embed and Send into Index:')
        for i in tqdm(range((n_docs // BATCH_SIZE) + 1)):
            batch_text = [doc.page_content for doc in docs[i*BATCH_SIZE: (i+1)*BATCH_SIZE]]
            metadatas = [{
                "chunk": j, "text": text, **metadata
            } for j, text in enumerate(batch_text)]

            ids = [str(uuid4()) for _ in range(len(batch_text))]

            logger.debug('Embedding...')
            embeds = embed.embed_documents(batch_text)
            logger.debug('Inserting into Index...')
            index.upsert(vectors=zip(ids, embeds, metadatas))


    logger.info(f'Index stats after: \n{index.describe_index_stats()}')


if __name__ == "__main__":
    create_index(index, './data/got-books')
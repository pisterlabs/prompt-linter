from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from lib import constants
import logging
import click

log = logging.getLogger(__name__)


def convert_scientsts_to_documents(list_data):
    title_db = set()
    result_list = []
    for data in list_data:
        tmp_data = data.copy()
        del tmp_data['page_content']
        del tmp_data['quote']
        title = data['title']
        if title not in title_db:
            title_db.add(title)
            click.secho(f'Adding {title}')
            doc = Document(page_content=data['page_content'],
                           metadata={**tmp_data})
            result_list.append(doc)
    return result_list


def populate(documents):  # pragma: no cover
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                   chunk_overlap=0)
    documents_splitted = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=constants.OPENAI_API_KEY)

    pinecone.init(
        api_key=constants.PINECONE_API_KEY,
        environment=constants.PINECONE_API_ENV,
    )

    index = pinecone.Index(constants.INDEX_NAME)
    index.delete(delete_all=True)
    click.secho('Inserting to pinecone')
    Pinecone.from_documents(documents_splitted,
                            embeddings,
                            index_name=constants.INDEX_NAME)

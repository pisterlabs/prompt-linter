try:
  import unzip_requirements # type: ignore
except ImportError:
  pass

import os
import logging
import pinecone

_logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "asia-southeast1-gcp-free")
ISSUE_REPO = os.getenv("ISSUE_REPO", "")

SIMIRARIRY_THRESHOLD = 0.85


def load_github_issues(repo=ISSUE_REPO):
    """Load GitHub issues from a repository into a list of strings."""
    from langchain.document_loaders import GitHubIssuesLoader

    loader = GitHubIssuesLoader(
        repo=repo,
        access_token=GITHUB_PERSONAL_ACCESS_TOKEN,
        state="all",
        include_prs=False, # only load issues
    )
    docs = loader.load()

    return docs


def setup_pinecone(index_name):
    # set up pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )


def generate_embeddings(docs):
    ## chunk data into smaller pieces
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    chunk_size =200

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    _logger.info(f'Now you have {len(texts)} documents')

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        client=None,
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    return text_splitter, embed


def create_index(index_name, docs, embed, text_splitter):
    from tqdm.auto import tqdm
    from uuid import uuid4

    index = pinecone.Index(index_name)
    index.describe_index_stats()
    batch_limit = 100

    data = docs
    texts = []
    metadatas = []

    for i, record in enumerate(tqdm(data)):
        # first get metadata fields for this record
        metadata = {
            'url': str(record.metadata['url']),
            'title': record.metadata['title'],
            'created_at': record.metadata['created_at'],
            'comments': record.metadata['comments'],
            'state': record.metadata['state'],
            'labels': record.metadata['labels'],
        }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record.page_content)
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas)) # type: ignore
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas)) # type: ignore
    
    return index


def delete_index(index_name):
    index = pinecone.Index(index_name)
    delete_response = index.delete(delete_all=True)
    return delete_response



def handler(event, context):
    _logger.info(event)

    # pinecone clientのセットアップ
    index_name = "error-defense-index"
    setup_pinecone(index_name)

    # github issuesを読み込み、embeddingを作成
    docs = load_github_issues()
    text_splitter, embed = generate_embeddings(docs)

    # pineconeの既存のindexを一旦flush
    delete_index(index_name)

    # embeddingから新規にindexを作成
    index = create_index(index_name, docs, embed, text_splitter)

    _logger.info(index.describe_index_stats())

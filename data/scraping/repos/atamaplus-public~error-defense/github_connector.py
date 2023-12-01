try:
  import unzip_requirements # type: ignore
except ImportError:
  pass

import os
import logging
from typing import Optional
import pinecone

from models.document import GithubDocument

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
_logger.addHandler(ch)

# env
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "asia-southeast1-gcp-free")
ISSUE_REPO = os.getenv("ISSUE_REPO", "")

SIMIRARIRY_THRESHOLD = 0.82


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


def query(index, embed, trace):
    def _unescape(s):
        return s.replace("\\n", "\n")

    from langchain.prompts import PromptTemplate
    from langchain.vectorstores import Pinecone

    text_field = "text"

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    template = """
    You will get the part of the stacktrace or error message below.
    If you get a stack trace, search for the Issue that contains the part of the stack trace that you think best captures the feature.
    If you get a short error message, search for an Issue that contains a message similar to the full text of the error message.

    What is the most similar issue below? :
    {stacktrace}
    """
    prompt = PromptTemplate(template=template, input_variables=["stacktrace"])
    query = prompt.format(stacktrace=_unescape(trace))

    answer = vectorstore.similarity_search_with_score(
        query,
        k=1
    )
    doc, score = answer[0]
    return (doc, score)
    


def run_query(trace) -> Optional[GithubDocument]:
    index_name = "error-defense-index"
    setup_pinecone(index_name)
    docs = load_github_issues()

    _, embed = generate_embeddings(docs)
    index = pinecone.Index(index_name)

    _logger.info(index.describe_index_stats())
    
    doc, score = query(index, embed, trace)
    _logger.info(f"doc: {doc}")
    _logger.info(f"score: {score}")

    if score < SIMIRARIRY_THRESHOLD:
        return None
    else:
        return GithubDocument(
            title = doc.metadata.get("title"),
            url = doc.metadata.get("url"),
            comments = doc.metadata.get("comments"),
            state = doc.metadata.get("state"),
            labels = doc.metadata.get("labels"),
            created_at = doc.metadata.get("created_at"),
        )

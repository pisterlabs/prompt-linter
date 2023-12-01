"""sage.viz module"""

import json
import os

import dotenv
import openai
import requests
from pydantic import BaseModel

from sage.document import Document

dotenv.load_dotenv()

github_token = os.getenv("API_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")
assert github_token, "API_TOKEN environment variable not set"
assert openai.api_key, "OPENAI_API_KEY environment variable not set"

gist_url = "https://api.github.com/gists"
gist_headers = {'Authorization': f'token {github_token}'}
gist_params = {'scope': 'gist'}

# Global variables
feature_vectors = []
metadata = []


class SearchQuery(BaseModel):
    searchText: str


def str_to_embeddings(text: str) -> list[float]:
    """
    Converts a paragraph to embeddings using OpenAI's text-embedding-ada-002 model.
    :param text: Text to embed
    :return: Embeddings of the paragraph
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings


def visualize(embeddings_vectors: list[list[float]], metadatas: list[Document]):
    """
    Visualize the embeddings vectors using the Tensorflow Projector.
    :param embeddings_vectors: List of embeddings vectors
    :param metadatas: List of metadata objects
    :return: URL to the Tensorflow Projector
    """

    feature_vecs = ""
    for embeddings_vector in embeddings_vectors:
        feature_vecs += "{}\n".format("\t".join([str(x) for x in embeddings_vector]))

    metadata = ("{}\t{}\t{}\t{}\t{}\t{}\n".format("Title", "URL", "Summary", "Authors", "DOI", "Date"))
    for i, paragraph in enumerate(metadatas):
        metadata += "{}\t{}\t{}\t{}\t{}\t{}\n".format(paragraph.name, paragraph.url, paragraph.summary,
                                                      paragraph.authors, paragraph.doi, paragraph.date)

    metadata_gist = upload_gist("metadata.tsv", metadata, "Metadata for vector visualization")
    feature_vecs_gist = upload_gist("feature_vec.tsv", feature_vecs, "Feature vectors for vector visualization")

    config = {
        'embeddings': [
            {
                'tensorName': 'embeddings',
                'tensorShape': [len(embeddings_vectors), len(embeddings_vectors[0])],
                'tensorPath': feature_vecs_gist,
                'metadataPath': metadata_gist,
            }
        ]
    }
    config_gist = upload_gist("config.json", json.dumps(config), "Config for vector visualization")
    return f'https://projector.tensorflow.org/?config={config_gist}'


def upload_gist(title, content, description):
    """
    Upload a gist to GitHub.
    :param title: Title of the gist
    :param content: Content of the gist
    :param description: Description of the gist
    :return: URL to the gist
    """
    payload = {"description": description, "public": True, "files": {title: {"content": content}}}
    res = requests.post(gist_url, headers=gist_headers, params=gist_params, data=json.dumps(payload))
    res = res.json()
    return res['files'][title]['raw_url']


def add_document(doc: Document):
    """
    Add a document to the list of documents.
    :param doc: Document to add
    :return: Status
    """
    embeddings = str_to_embeddings(doc.summary)
    feature_vectors.append(embeddings)
    metadata.append(doc)
    return {"status": "ok"}


def add_multiple_documents(docs: list[Document]):
    """
    Add multiple documents to the list of documents.
    :param docs: List of documents
    :return: Status
    """
    for page in docs:
        embeddings = str_to_embeddings(page.summary)
        feature_vectors.append(embeddings)
        metadata.append(page)
    return {"status": "ok"}


def visualize_handler():
    """Visualize the documents."""
    url = visualize(feature_vectors, metadata)
    print({"url": url})
    return {"url": url}

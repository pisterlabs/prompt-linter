import datetime
import requests
from typing import Dict, List, Any

from langchain.schema.document import Document
from langchain.schema.vectorstore import VectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import (
    SemanticSimilarityExampleSelector
)
from langchain.pydantic_v1 import Extra
from langchain.vectorstores import Pinecone

from stampy_chat.env import PINECONE_INDEX, PINECONE_NAMESPACE, OPENAI_API_KEY, REMOTE_CHAT_INSTANCE
from stampy_chat.callbacks import StampyCallbackHandler



class RemoteVectorStore(VectorStore):
    """Make a wrapper around the deployed semantic search.

    One of the prerequisites for the chat bot to work properly is for Pinecone to be set up. This can
    be a bother, so to make things easier, this will just call the semantic search of the deployed chatbot.
    """

    def add_texts(self, *args, **kwargs):
        "This is an abstract method, so must be instanciated..."

    def from_texts(self, *args, **kwargs):
        "This is an abstract method, so must be instanciated..."

    def similarity_search(self, *args, **kwargs):
        "This is an abstract method, so must be instanciated..."

    def similarity_search_with_score(self, query, k=2, **kwargs):
        results = requests.post(REMOTE_CHAT_INSTANCE + "/semantic", json={'query': query, 'k': k}).json()
        SCORE = 1 # set the score to 1, as it's already been filtered once
        return [
            (Document(page_content=res.get('id'), metadata=dict(res, date_published=res.get('date'))), SCORE)
            for res in results
        ]


class ReferencesSelector(SemanticSimilarityExampleSelector):
    """Get examples with enumerated indexes added."""

    callbacks: List[StampyCallbackHandler] = []
    history_field: str = 'history'
    min_score: float = 0.8  # docs with lower scores will be excluded from the context

    class Config:
        """This is needed for extra fields to be added... """
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @staticmethod
    def make_reference(i: int) -> str:
        """Make the reference used in citations - basically translate i -> 'a + i'"""
        return chr(i + 97)

    def fetch_docs(self, input_variables) -> List:
        ### Copied from parent - for some reason they ignore the ids of the returned items, so
        # it has to be added manually here...
        if self.input_keys:
            input_variables = {key: input_variables[key] for key in self.input_keys}
        query = " ".join(v for v in input_variables.values())
        example_docs = [
            doc for doc, score in self.vectorstore.similarity_search_with_score(query, k=self.k)
            if score > self.min_score
        ]

        # Remove any duplicates - sometimes the same document is returned multiple times
        return list({e.page_content: e for e in example_docs}.values())

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Fetch the top matching items from the underlying storage and add indexes.

        :param Dict[str, str] input_variables: a dict of {<field>: <query>} pairs to look through the dataset
        :returns: a list of example objects
        """
        for callback in self.callbacks:
            callback.on_context_fetch_start(input_variables)

        input_variables = dict(**input_variables)
        history = input_variables.pop(self.history_field, [])

        examples = self.fetch_docs(input_variables)

        for item in history[::-1]:
            if len(examples) >= self.k:
                break
            examples += self.fetch_docs({'answer': item.content})

        examples = [
            dict(
                e.metadata,
                id=e.page_content,
                reference=self.make_reference(i)
            ) for i, e in enumerate(examples)
        ]

        for callback in self.callbacks:
            callback.on_context_fetch_end(examples)

        return examples


def make_example_selector(**params) -> ReferencesSelector:
    if PINECONE_INDEX:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Pinecone(PINECONE_INDEX, embeddings.embed_query, "hash_id", namespace=PINECONE_NAMESPACE)
    else:
        vectorstore = RemoteVectorStore()
    return ReferencesSelector(vectorstore=vectorstore, **params)


def format_block(block) -> Dict[str, Any]:
    date = block.get('date_published') or block.get('date')

    if isinstance(date, datetime.datetime):
        date = date.date().isoformat()
    elif isinstance(date, datetime.date):
        date = date.isoformat()
    elif isinstance(date, (int, float)):
        date = datetime.datetime.fromtimestamp(date).date().isoformat()

    authors = block.get('authors')
    if not authors and block.get('author'):
        authors = [block.get('author')]

    return {
        "id": block.get('hash_id') or block.get('id'),
        "title": block['title'],
        "authors": authors,
        "date": date,
        "url": block['url'],
        "tags": block.get('tags'),
        "text": block['text']
    }


def get_top_k_blocks(query, k):
    blocks = make_example_selector(k=k).select_examples({'query': query})
    return list(map(format_block, blocks))

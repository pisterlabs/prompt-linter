"""Split documents using upon a trained model."""

import os
import pickle
from typing import Any
from typing import Sequence

import openai
import spacy
from langchain.schema.document import BaseDocumentTransformer
from langchain.schema.document import Document
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm.autonotebook import tqdm

from models.split_model_train import predict_using_features_and_ensemble
from models.split_model_train import predict_using_features_and_greedy_embeddings
from models.split_model_train import syntactic_paragraph_features
from models.split_utils import get_mpnet_embedder
from models.split_utils import get_openai_embedder
from models.split_utils import get_paragraph_sentence_texts_and_ids
from models.split_utils import get_paragraph_texts_and_ids
from models.split_utils import get_split_texts_and_ids
from models.split_utils import get_voyageai_embedder
from models.split_utils import split_on_markdown_headers


def split_document_content(
    page_content: str, metadata: dict[str, Any], parser: Any, splitter: Any, max_chars: int, anchor: str
) -> list[Document]:
    """Split document content into chunks."""
    docs = []
    # get paragraphs/sentences texts and ids
    paragraph_texts_and_ids = get_paragraph_sentence_texts_and_ids(page_content, parser, max_chars)
    paragraphs = [paragraph_text_id[0] for paragraph_text_id in paragraph_texts_and_ids]
    # get splits
    splits = splitter(paragraphs)
    split_texts_and_ids = get_split_texts_and_ids(
        paragraph_texts_and_ids,
        splits,
    )
    # create splits
    for split_text_and_id in split_texts_and_ids:
        if anchor:
            metadata = metadata.copy()
            metadata[anchor] = split_text_and_id[1]
        else:
            metadata = metadata
        docs.append(Document(metadata=metadata, page_content=split_text_and_id[0]))
    return docs


class SyntacticEmbeddingSplitter(BaseDocumentTransformer):
    """Split documents recursively, then join them based upon syntax and embedding similarity."""

    def __init__(
        self,
        embedder: Any = None,
        split_threshold: float = 0.83,
        max_chars: int = 2000,
        anchor: str = "anchor",
        **kwargs: Any
    ):
        """Initialize."""
        super().__init__(**kwargs)

        self.max_chars = max_chars
        self.anchor = anchor
        # init spacy
        self.parser = spacy.load("en_core_web_sm")

        if embedder is None:
            embedder = get_voyageai_embedder()

        self.splitter = predict_using_features_and_greedy_embeddings(
            syntactic_paragraph_features, embedder, split_threshold, max_chars
        )

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Transform documents by splitting them and then joining the splits using syntactic features and embeddings."""
        verbose = kwargs.get("verbose", False)
        transformed_docs: list[Document] = []
        for doc in tqdm(documents, disable=not verbose):
            # split each document
            transformed_docs.extend(
                split_document_content(
                    doc.page_content,
                    doc.metadata,
                    self.parser,
                    self.splitter,
                    self.max_chars,
                    self.anchor,
                )
            )
        return transformed_docs

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Transform documents asynchronously."""
        raise NotImplementedError

    def split_documents(self, documents: Sequence[Document], verbose: bool = False) -> list[Document]:
        """Split documents using model."""
        return list(self.transform_documents(documents, verbose=verbose))


class MarkdownSyntacticEmbeddingSplitter(SyntacticEmbeddingSplitter):
    """Split documents recursively, then join them based upon syntax and embedding similarity."""

    def __init__(
        self,
        embedder: Any = None,
        split_threshold: float = 0.83,
        max_chars: int = 2000,
        anchor: str = "anchor",
        header_separator: str = " / ",
        **kwargs: Any
    ):
        """Initialize by calling SyntacticEmbeddingSplitter."""
        super().__init__(embedder, split_threshold, max_chars, anchor, **kwargs)
        self.header_separator = header_separator

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Transform documents by splitting them into sections and then splitting each section."""
        verbose = kwargs.get("verbose", False)
        transformed_docs: list[Document] = []
        for doc in tqdm(documents, disable=not verbose):
            # get markdown sections
            for section, headers in split_on_markdown_headers(doc.page_content, self.max_chars):
                # split each section
                metadata = doc.metadata.copy()
                if self.header_separator and len(headers) > 0:
                    metadata["title"] += self.header_separator + self.header_separator.join(headers)
                else:
                    section = " ".join(headers) + "\n\n" + section
                transformed_docs.extend(
                    split_document_content(
                        section,
                        metadata,
                        self.parser,
                        self.splitter,
                        self.max_chars,
                        self.anchor,
                    )
                )
        return transformed_docs


class ModelTextSplitter(BaseDocumentTransformer):
    """Split documents using upon a trained model."""

    def __init__(
        self,
        model_path: str = "",
        split_threshold: float = 0.55,
        chunk_size: int = 500,
        anchor: str = "anchor",
        **kwargs: Any
    ):
        """Initialize splitter with model."""
        super(BaseDocumentTransformer, self).__init__(**kwargs)

        self.chunk_size = chunk_size
        self.anchor = anchor

        # load model
        with open(model_path, "rb") as f:  # nosec B301
            clf = pickle.load(f)

        # init spacy
        parser = spacy.load("en_core_web_sm")

        # load mpnet embedder for splitter
        mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        mpnet_embedder = get_mpnet_embedder(mpnet)

        # openai embedder for splitter
        openai.api_key = os.environ["OPENAI_API_KEY"]
        openai_embedder = get_openai_embedder(openai)

        self.predictor = predict_using_features_and_ensemble(
            syntactic_paragraph_features,
            openai_embedder,
            mpnet_embedder,
            parser,
            clf,
            split_threshold,
        )

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Transform documents by splitting them using model."""
        verbose = kwargs.get("verbose", False)
        transformed: list[Document] = []
        for doc in tqdm(documents, disable=not verbose):
            # get paragraphs
            paragraph_texts_and_ids = get_paragraph_texts_and_ids(doc.page_content)
            paragraphs = [paragraph_text_id[0] for paragraph_text_id in paragraph_texts_and_ids]

            # get splits
            splits = self.predictor(paragraphs)
            split_texts_and_ids = get_split_texts_and_ids(
                paragraph_texts_and_ids,
                splits,
                max_split_len=self.chunk_size,
            )

            # create split
            for split_text_and_id in split_texts_and_ids:
                if self.anchor:
                    metadata = doc.metadata.copy()
                    metadata[self.anchor] = split_text_and_id[1]
                else:
                    metadata = doc.metadata
                transformed.append(Document(metadata=metadata, page_content=split_text_and_id[0]))
        return transformed

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Transform documents asynchronously."""
        raise NotImplementedError

    def split_documents(self, documents: Sequence[Document], verbose: bool = False) -> list[Document]:
        """Split documents using model."""
        return list(self.transform_documents(documents, verbose=verbose))

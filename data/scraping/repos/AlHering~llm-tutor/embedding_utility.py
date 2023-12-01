# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, List, Tuple
import torch.nn.functional as F
from torch import Tensor
from chromadb.api.types import EmbeddingFunction, Documents
from chromadb.api.types import Embeddings as CDBEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings as LCEmbeddings
from typing import List
from sentence_transformers import SentenceTransformer
from ..bronze import json_utility


class LocalHuggingFaceEmbeddings(LCEmbeddings):
    """
    Class for using local sentence transformer models from Huggingface as embeddings.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initiation method.
        :param model_path: Model path.
        """
        try:
            self.embedding_model = SentenceTransformer(model_path)
        except TypeError as ex:
            if str(ex) == "Pooling.__init__() got an unexpected keyword argument 'pooling_mode_weightedmean_tokens'":
                print(
                    "Encountered error (https://huggingface.co/hkunlp/instructor-base/discussions/6), adjusting local files.")
                print(
                    f"Try 'pip install --force --no-deps git+https://github.com/UKPLab/sentence-transformers.git'")
                raise ex

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Method for embedding document texts.
        :param texts: Texts.
        :return: A list of embeddings for each text in the form of a list of floats.
        """
        return [embedding.tolist() for embedding in self.embedding_model.encode(texts)]

    def embed_query(self, query: str) -> List[float]:
        """
        Method for embedding a query.
        :param query: Query.
        :return: The embedding for the given queryin the form of a list of floats.
        """
        return self.embedding_model.encode([query])[0].tolist()


class T5EmbeddingFunction(EmbeddingFunction):
    """
    EmbeddingFunction utilizing the "intfloat_e5-large-v2" model.
    (https://huggingface.co/intfloat/e5-large-v2)
    """

    def __init__(self, model_path: str) -> None:
        """
        Initiation method.
        :param model_path: Model path.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True)

    def embed_documents(self, texts: Documents) -> CDBEmbeddings:
        """
        Method handling embedding.
        :param texts: Texts to embed.
        """
        # Taken from https://huggingface.co/intfloat/e5-large-v2 and adjusted
        # Tokenize the input texts
        batch_dict = self.tokenizer(texts, max_length=512,
                                    padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state,
                                       batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def embed_query(self, query: str) -> CDBEmbeddings:
        """
        Method for embedding query.
        :param query: Query.
        :return: Query embedding.
        """
        batch_dict = self.tokenizer(query, max_length=512,
                                    padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state,
                                       batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Average pooling function, taken from https://huggingface.co/intfloat/e5-large-v2.
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

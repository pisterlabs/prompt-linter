import sys
import os
from typing import List
import numpy

from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config import TEXTENCODER_CONFIG  # pylint: disable=C0413

MODEL = TEXTENCODER_CONFIG.get('model', 'multi-qa-mpnet-base-cos-v1')
NORM = TEXTENCODER_CONFIG.get('norm', False)


class TextEncoder(HuggingFaceEmbeddings):
    '''Text encoder converts text input(s) into embedding(s)'''

    def __init__(self, *args, **kwargs):
        assert isinstance(
            self, Embeddings), 'Invalid text encoder. Only accept LangChain embeddings.'
        kwargs['model_name'] = kwargs.get('model_name', MODEL)
        super().__init__(*args, **kwargs)

    def embed_documents(self, texts: List[str], norm: bool = NORM) -> List[List[float]]:
        embeds = super().embed_documents(texts)
        if norm:
            embeds = [(x / numpy.linalg.norm(x)).tolist() for x in embeds]
        return embeds

    def embed_query(self, text: str, norm: bool = NORM) -> List[float]:
        embed = super().embed_query(text)
        if norm:
            embed /= numpy.linalg.norm(embed)
            embed = embed.tolist()
        return embed

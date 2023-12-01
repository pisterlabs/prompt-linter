from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, TypeVar
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores.base import VectorStore

VST = TypeVar("VST", bound="VectorStore")


@dataclass
class QAData:
    vst: VST
    documents: List[Document]
    chain: RetrievalQAWithSourcesChain

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # eval
#
# > 

# %%
#| default_exp eval

# %%
#| hide
from nbdev.showdoc import *

# %%
#| export
from abc import ABC
from abc import abstractmethod
from typing import Literal
from typing import TypeVar
import os

from emblem.data import Chunks
from annoy import AnnoyIndex as _AnnoyIndex
from emblem.core import A
from emblem.core import AT
from emblem.api import OpenAIModel
from emblem.model import EmbeddingModel
import pandas as pd
from emblem.llm import completion

# %%
#| export
I = TypeVar("I", bound=_AnnoyIndex)


# %%
#| export
class Index(ABC):
    def __init__(
        self, 
        path: str | None = None,
        text: Chunks | None = None,
        model: EmbeddingModel | None = None,
        size: int | None = None,
        metric: str | None = None,
    ) -> None:
        self.model = model
        self.size = size
        self.metric = metric
        
        if self.model is None:
            self.model = OpenAIModel() 

        if self.size is None:
            test_text = "This is demo text."
            self.size = len(self.model.embed(test_text))

        if path is None:
            self.index = self._create_index(text, self.model, self.size, self.metric)
        else:
            self.index = self.load(path, self.size, self.metric)

    @abstractmethod
    def _create_index(self, text: str, model: EmbeddingModel, metric: str) -> I:
        ...

    @abstractmethod
    def search(self, text: str, n: int = 1, type: AT = "float") -> tuple[list[int], list[A]]:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str, size: int, metric: str | None) -> I:
        ...


# %%
path = "/Users/mvkvc/dev/emblem/data/case_brief.pdf"
os.path.dirname(path)


# %%
#| export
class AnnoyIndex(Index):
    def _create_index(self, text: str, model: EmbeddingModel, metric: str = "angular") -> I:
        index = _AnnoyIndex(size, metric)

        for id, s in text:
            embedding = model.embed(s)
            index.add_item(id, embedding)

        n_trees = 10
        index.build(n_trees)
        
        return index

    def search(self, text: str, n: int = 1, type: AT = "float") -> list[tuple[int, list[A]]]:
        vector = self.model.embed(text)
        results = self.index.get_nns_by_vector(v, n, search_k=-1, include_distances=True)
        # If additional changing needed do here (type conversion for AT)
        return results
        
    def save(self, path: str) -> None:
        dir = os.path.dirname(path)
        
        if os.path.exists(dir):
            self.index.save(path)
        else:
            raise ValueError("Invalid path selected.")
            
    def load(self, path: str, size: int, metric: str | None) -> I:
        metric = metric if metric else "angular"
        index = _AnnoyIndex(size, metric)

        if os.path.exists(path):
            index.load(path)
        else:
            raise ValueError("Invalid path selected.")

        return index



# %%
#| export
class Eval(ABC):
    def __init__(self, models: list[EmbeddingModel]) -> None:
        self.models = models
        
    @abstractmethod
    def run(self, **kwargs) -> list[tuple[str, float]]:
        ...

    @abstractmethod
    def plot(self, path: str | None = None) -> None:
        ...


# %%
#| export
class Questions:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    @classmethod
    def generate(cls, chunks: Chunks, prompt: str | None = None ) -> 'Questions':
        df = chunks.data
        #

    @classmethod
    def load(cls, path: str) -> 'Questions':
        data = pd.read_csv(path)
        
        return cls(data)

    def save(self, path: str) -> None:
        self.data.to_csv(path)


# %%
#| export
class RAGEval(Eval):
    def run(
            self,
            path_chunks: str,
            path_questions: str | None = None,
            n: int = 5
        ):
        self.path_chunks = path_chunks
        self.path_questions = path_questions
        
        self.chunks = Chunks.load(self.path_chunks)
        
        if self.path_questions is not None:
            self.questions = pd.load_csv(path_questions)
        else:
            self.questions = _generate_questions(self.chunks)

        #


# %%
# | hide
import nbdev

nbdev.nbdev_export()

# %%

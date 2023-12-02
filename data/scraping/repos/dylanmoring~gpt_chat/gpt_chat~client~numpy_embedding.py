from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np

from .open_ai_client import OpenAIClient


class NumpyEmbeddingOpenAIClient(OpenAIClient):
    @classmethod
    async def get_embedding(cls, text: str) -> Optional[np.ndarray]:
        embedding = await super().get_embedding(text)
        return np.array(embedding)

    @classmethod
    async def get_embeddings(cls, list_of_text: List[str]) -> List[np.ndarray]:
        embeddings = await super().get_embeddings(list_of_text)
        return [np.array(embedding) for embedding in embeddings]

    @classmethod
    async def get_embeddings_column(cls, text_column: pd.Series) -> pd.Series:
        embeddings = await cls.get_embeddings(text_column.values.tolist())
        return pd.Series(
            data=embeddings,
            index=text_column.index,
            name='embedding'
        )

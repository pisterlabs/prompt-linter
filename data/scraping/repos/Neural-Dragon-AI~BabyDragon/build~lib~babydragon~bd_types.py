from enum import Enum
from typing import Callable, Tuple

import numpy as np

from babydragon.models.embedders.ada2 import OpenAiEmbedder


class EmbeddableType(Enum):
    TEXT = "text"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    # Add more data types as required


def infer_embeddable_type(column) -> Tuple[EmbeddableType, Callable]:
    # Infer the data type of the column
    # This will depend on the type of `column` (whether it's a string, Series, etc.)
    # Here we'll assume `column` is a pandas Series for simplicity
    column_type = str(column.dtype)
    print(column_type)
    if column_type == "Utf8":
        # If it's an object, we'll assume it's text
        return EmbeddableType.TEXT, OpenAiEmbedder()
    elif np.issubdtype(column.dtype, np.number):
        # If it's a number, we'll use a different embedding strategy
        return EmbeddableType.NUMERIC, numeric_embedder
    else:
        # For other types, we could throw an error or have a default strategy
        raise ValueError(f"Cannot infer type for column {column.name}")


def numeric_embedder(column):
    # Implement the numeric embedding strategy
    # This will depend on the type of `column` (whether it's a string, Series, etc.)
    # Here we'll assume `column` is a pandas Series for simplicity
    return column.values

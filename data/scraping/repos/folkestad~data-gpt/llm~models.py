import dataclasses

import guidance
from chromadb.utils import embedding_functions


@dataclasses.dataclass(frozen=True)
class Models:
    OPENAI: str = "openai"
    NONE: str = "none"


def llm_model(model_type: str, llm_model_name: str, api_key: str | None = None):
    if model_type == Models.OPENAI:
        return guidance.llms.OpenAI(model=llm_model_name, api_key=api_key)
    else:
        raise ValueError(f"Unknown llm model type: {model_type}")


def embedding_model(
    model_type: str, embedding_model_name: str, api_key: str | None = None
):
    if model_type == Models.OPENAI:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=embedding_model_name
        )
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")


def embedding_model_default():
    return embedding_functions.DefaultEmbeddingFunction()

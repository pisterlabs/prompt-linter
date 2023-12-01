import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
from functools import wraps

from openai import OpenAI

from langchain.embeddings import HuggingFaceEmbeddings
from openai import (
    OpenAIError,
    RateLimitError,
    APIError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    retry_if_exception,
)

from align_data.embeddings.pinecone.pinecone_models import MissingEmbeddingModelError
from align_data.settings import (
    USE_OPENAI_EMBEDDINGS,
    OPENAI_EMBEDDINGS_MODEL,
    OPENAI_API_KEY,
    OPENAI_ORGANIZATION,
    EMBEDDING_LENGTH_BIAS,
    SENTENCE_TRANSFORMER_EMBEDDINGS_MODEL,
    DEVICE,
)

client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORGANIZATION)

# --------------------
# CONSTANTS & CONFIGURATION
# --------------------

logger = logging.getLogger(__name__)

hf_embedding_model = None
if not USE_OPENAI_EMBEDDINGS:
    hf_embedding_model = HuggingFaceEmbeddings(
        model_name=SENTENCE_TRANSFORMER_EMBEDDINGS_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"show_progress_bar": False},
    )

ModerationInfoType = Dict[str, Any]


# --------------------
# DECORATORS
# --------------------


def handle_openai_errors(func):
    """Decorator to handle OpenAI-specific exceptions with retries."""

    @wraps(func)
    @retry(
        wait=wait_random_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIError)
        | retry_if_exception(lambda e: "502" in str(e)),
    )
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            logger.warning(f"OpenAI Rate limit error. Trying again. Error: {e}")
            raise
        except APIError as e:
            if "502" in str(e):
                logger.warning(f"OpenAI 502 Bad Gateway error. Trying again. Error: {e}")
            else:
                logger.error(f"OpenAI API Error encountered: {e}")
            raise
        except OpenAIError as e:
            logger.error(f"OpenAI Error encountered: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error encountered: {e}")
            raise

    return wrapper


# --------------------
# MAIN FUNCTIONS
# --------------------


@handle_openai_errors
def _single_batch_moderation_check(batch: List[str]) -> List[ModerationInfoType]:
    """Process a batch for moderation checks."""
    return client.moderations.create(input=batch)["results"]


def moderation_check(texts: List[str], max_batch_size: int = 4096, tokens_counter: Callable[[str], int] = len) -> List[ModerationInfoType]:
    """Batch moderation checks on list of texts.

    :param List[str] texts: the texts to be checked
    :param int max_batch_size: the max size in tokens for a single batch
    :param Callable[[str], int] tokens_counter: the function used to count tokens
    """
    # A very ugly loop that will split the `texts` into smaller batches so that the
    # total sum of tokens in each batch will not exceed `max_batch_size`
    parts = []
    part = []
    part_count = 0
    for item in texts:
        if part_count + tokens_counter(item) > max_batch_size:
            parts.append(part)
            part = []
            part_count = 0
        part.append(item)
        part_count += tokens_counter(item)
    if part:
        parts.append(part)

    return [
        result
        for batch in parts
        for result in _single_batch_moderation_check(batch)
    ]


@handle_openai_errors
def _single_batch_compute_openai_embeddings(batch: List[str], **kwargs) -> List[List[float]]:
    """Compute embeddings for a batch."""
    batch_data = client.embeddings.create(input=batch, engine=OPENAI_EMBEDDINGS_MODEL, **kwargs).data
    return [d["embedding"] for d in batch_data]


def _compute_openai_embeddings(
    non_flagged_texts: List[str], max_texts_num: int = 2048, **kwargs
) -> List[List[float]]:
    """Batch computation of embeddings for non-flagged texts."""
    return [
        embedding
        for batch in (
            non_flagged_texts[i : i + max_texts_num]
            for i in range(0, len(non_flagged_texts), max_texts_num)
        )
        for embedding in _single_batch_compute_openai_embeddings(batch, **kwargs)
    ]


def get_embeddings_without_moderation(
    texts: List[str],
    source: Optional[str] = None,
    **kwargs,
) -> List[List[float]]:
    """
    Obtain embeddings without moderation checks.

    Parameters:
    - texts (List[str]): List of texts to be embedded.
    - source (Optional[str], optional): Source identifier to potentially adjust embedding bias. Defaults to None.
    - **kwargs: Additional keyword arguments passed to the embedding function.

    Returns:
    - List[List[float]]: List of embeddings for the provided texts.
    """
    if not texts:
        return []

    texts = [text.replace("\n", " ") for text in texts]
    if USE_OPENAI_EMBEDDINGS:
        embeddings = _compute_openai_embeddings(texts, **kwargs)
    elif hf_embedding_model:
        embeddings = hf_embedding_model.embed_documents(texts)
    else:
        raise MissingEmbeddingModelError("No embedding model available.")

    # Bias adjustment
    if source and (bias := EMBEDDING_LENGTH_BIAS.get(source, 1.0)):
        embeddings = [[bias * e for e in embedding] for embedding in embeddings]

    return embeddings


def get_embeddings_or_none_if_flagged(
    texts: List[str],
    source: Optional[str] = None,
    **kwargs,
) -> Tuple[List[List[float]] | None, List[ModerationInfoType]]:
    """
    Obtain embeddings for the provided texts. If any text is flagged during moderation,
    the function returns None for the embeddings while still providing the moderation results.

    Parameters:
    - texts (List[str]): List of texts to be embedded.
    - source (Optional[str], optional): Source identifier to potentially adjust embedding bias. Defaults to None.
    - **kwargs: Additional keyword arguments passed to the embedding function.

    Returns:
    - Tuple[Optional[List[List[float]]], ModerationInfoListType]: Tuple containing the list of embeddings (or None if any text is flagged) and the moderation results.
    """
    moderation_results = moderation_check(texts)
    if any(result["flagged"] for result in moderation_results):
        return None, moderation_results

    embeddings = get_embeddings_without_moderation(texts, source, **kwargs)
    return embeddings, moderation_results


def get_embeddings(
    texts: List[str],
    source: Optional[str] = None,
    **kwargs,
) -> Tuple[List[List[float] | None], List[ModerationInfoType]]:
    """
    Obtain embeddings for the provided texts, replacing the embeddings of flagged texts with `None`.

    Parameters:
    - texts (List[str]): List of texts to be embedded.
    - source (Optional[str], optional): Source identifier to potentially adjust embedding bias. Defaults to None.
    - **kwargs: Additional keyword arguments passed to the embedding function.

    Returns:
    - Tuple[List[Optional[List[float]]], ModerationInfoListType]: Tuple containing the list of embeddings (with None for flagged texts) and the moderation results.
    """
    assert all(texts), "No empty strings allowed in the input list."

    # replace newlines, which can negatively affect performance
    texts = [text.replace("\n", " ") for text in texts]

    # Check all texts for moderation flags
    moderation_results = moderation_check(texts)
    flags = [result["flagged"] for result in moderation_results]

    non_flagged_texts = [text for text, flag in zip(texts, flags) if not flag]
    non_flagged_embeddings = get_embeddings_without_moderation(non_flagged_texts, source, **kwargs)
    embeddings = [None if flag else non_flagged_embeddings.pop(0) for flag in flags]
    return embeddings, moderation_results


def get_embedding(
    text: str, source: Optional[str] = None, **kwargs
) -> Tuple[List[float] | None, ModerationInfoType]:
    """Obtain an embedding for a single text."""
    embedding, moderation_result = get_embeddings([text], source, **kwargs)
    return embedding[0], moderation_result[0]

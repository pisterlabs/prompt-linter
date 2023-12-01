from typing import TypeVar, Union, List, Mapping, Sequence, Any, Tuple
from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings

T = TypeVar("T")

OneOrMany = Union[T, List[T]]
ID = str
IDs = List[ID]
Embedding = List[float]
Metadata = Mapping[str, Union[str, int, float, bool]]
Metadatas = List[Metadata]

Parameter = TypeVar("Parameter", Embedding, Document, Metadata, ID)


def maybe_cast_one_to_many(
    target: OneOrMany[Parameter],
) -> List[Parameter]:
    """Infers if target is Embedding, Metadata, or Document and casts it to a many object if its one"""

    if isinstance(target, Sequence):
        # One Document or ID
        if isinstance(target, str) and target is not None:
            return [target]
        # One Embedding
        if isinstance(target[0], (int, float)):
            return [target]  # type: ignore
    # One Metadata dict
    if isinstance(target, dict):
        return [target]
    # Already a sequence
    return target  # type: ignore


def validate_embeddings(embeddings: Embeddings) -> Embeddings:
    """Validates embeddings to ensure it is a list of list of ints, or floats"""
    if not isinstance(embeddings, list):
        raise ValueError(f"Expected embeddings to be a list, got {embeddings}")
    if len(embeddings) == 0:
        raise ValueError(
            f"Expected embeddings to be a list with at least one item, got {embeddings}"
        )
    if not all([isinstance(e, list) for e in embeddings]):
        raise ValueError(
            f"Expected each embedding in the embeddings to be a list, got {embeddings}"
        )
    for embedding in embeddings:
        if not all([isinstance(value, (int, float)) for value in embedding]):
            raise ValueError(
                f"Expected each value in the embedding to be a int or float, got {embeddings}"
            )
    return embeddings


def validate_ids(ids: IDs) -> IDs:
    """Validates ids to ensure it is a list of strings"""
    if not isinstance(ids, list):
        raise ValueError(f"Expected IDs to be a list, got {ids}")
    if len(ids) == 0:
        raise ValueError(f"Expected IDs to be a non-empty list, got {ids}")
    seen = set()
    dups = set()
    for id_ in ids:
        if not isinstance(id_, str):
            raise ValueError(f"Expected ID to be a str, got {id_}")
        if id_ in seen:
            dups.add(id_)
        else:
            seen.add(id_)
    if dups:
        n_dups = len(dups)
        if n_dups < 10:
            example_string = ", ".join(dups)
            message = (
                f"Expected IDs to be unique, found duplicates of: {example_string}"
            )
        else:
            examples = []
            for idx, dup in enumerate(dups):
                examples.append(dup)
                if idx == 10:
                    break
            example_string = (
                f"{', '.join(examples[:5])}, ..., {', '.join(examples[-5:])}"
            )
            message = f"Expected IDs to be unique, found {n_dups} duplicated IDs: {example_string}"
        raise ValueError(message)
    return ids


def validate_metadata(metadata: Metadata) -> Metadata:
    """Validates metadata to ensure it is a dictionary of strings to strings, ints, floats or bools"""
    if not isinstance(metadata, dict) and metadata is not None:
        raise ValueError(f"Expected metadata to be a dict or None, got {metadata}")
    if metadata is None:
        return metadata
    if len(metadata) == 0:
        raise ValueError(f"Expected metadata to be a non-empty dict, got {metadata}")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Expected metadata key to be a str, got {key} which is a {type(key)}"
            )
        # isinstance(True, int) evaluates to True, so we need to check for bools separately
        if not isinstance(value, bool) and not isinstance(value, (str, int, float)):
            raise ValueError(
                f"Expected metadata value to be a str, int, float or bool, got {value} which is a {type(value)}"
            )
    return metadata


def validate_metadatas(metadatas: Metadatas) -> Metadatas:
    """Validates metadatas to ensure it is a list of dictionaries of strings to strings, ints, floats or bools"""
    if not isinstance(metadatas, list):
        raise ValueError(f"Expected metadatas to be a list, got {metadatas}")
    for metadata in metadatas:
        validate_metadata(metadata)
    return metadatas


def _results_to_docs(results: Any) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

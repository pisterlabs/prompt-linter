import numpy as np

from langchain.schema.embeddings import Embeddings
from ragflow.commons.chroma import ChromaClient
from ragflow.commons.configurations import BaseConfigurations

import logging

logger = logging.getLogger(__name__)


def grade_embedding_similarity(
    label_dataset: list[dict],
    predictions: list[dict],
    embedding_model: Embeddings,
    user_id: str,
) -> float:
    """Calculate similarities of label answers and generated answers using the corresponding embeddings. We multiply the matrixes of the provided embeddings and take the average of the diagonal values, which should be the cosine similarites assuming that the embeddings were already normalized.

    Args:
        label_dataset (list[dict]): The evaluation ground truth dataset of QA pairs
        predictions (list[dict]): A dict containing the predicted answers from the queries and the retrieved document chunks
        embedding_model (Embeddings): The embedding model
        user_id (str): The user id, probably in UUID format, used to query the embeddings of the answers from the evaluation dataset from ChromaDB.

    Returns:
        float: The average similarity score.
    """
    logger.info("Calculating embedding similarities.")

    num_qa_pairs = len(label_dataset)

    label_answers = [qa_pair["answer"] for qa_pair in label_dataset]
    predicted_answers = [qa_pair["result"] for qa_pair in predictions]

    # try using embeddings of answers of evaluation set from vectorstore that were stored in ChromaDB in generation process, otherwise we calculate them again
    try:
        with ChromaClient() as CHROMA_CLIENT:
            collection_id = f"userid_{user_id}_qaid_0_{BaseConfigurations.get_embedding_model_name(embedding_model)}"

            for col in CHROMA_CLIENT.list_collections():
                if col.metadata.get("custom_id", "") == collection_id:
                    collection = col
                    break

            ids = [qa["metadata"]["id"] for qa in label_dataset]

            target_embeddings = np.array(
                collection.get(ids=ids, include=["embeddings"])["embeddings"]
            ).reshape(num_qa_pairs, -1)

            logger.info("Embeddings for label answers loaded successfully.")

    except Exception as ex:
        logger.info(
            f"Embeddings of {BaseConfigurations.get_embedding_model_name(embedding_model)} for label answers could not be loaded from vectorstore.\n\
            Collections: {CHROMA_CLIENT.list_collections()}.\n\
            Exception: {ex.args}"
        )

        target_embeddings = np.array(
            embedding_model.embed_documents(label_answers)
        ).reshape(num_qa_pairs, -1)

    predicted_embeddings = np.array(
        embedding_model.embed_documents(predicted_answers)
    ).reshape(num_qa_pairs, -1)

    emb_norms = np.linalg.norm(target_embeddings, axis=1) * np.linalg.norm(
        predicted_embeddings, axis=1
    )

    dot_prod = np.diag(np.dot(target_embeddings, predicted_embeddings.T))

    similarities = dot_prod / emb_norms

    return np.nanmean(similarities)

import qdrant_client, grpc
from langchain.vectorstores import Qdrant
import config
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from typing import Any, List, Optional
from pathlib import Path

def load_embeddings(model_name:str = config.EMBBEDINGS_MODEL ) -> Any:
    """ Load HuggingFaceEmbeddings
    Args:
        model_name (str, optional): Name of model. Defaults to `config.EMBBEDINGS_MODEL`.
        
    Returns:
        Any: embeddings model
    """    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def connect_db(collection:str, 
               distance_strategy:str = "COSINE",
               embeddings = load_embeddings()) -> Qdrant:
    """ Connect to Qdrant collection

    Args:
        collection (str): Name of collection
        distance_strategy (str, optional): Distance Estrategy #EUCLID, #COSINE, #DOT. Defaults to `COSINE`.
        embeddings (Any, optional): Embeddings model. Defaults to `load_embeddings()`.
        
    Returns:
        Qdrant: Qdrant Instance
    """    

    client = qdrant_client.QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_HOST_PORT,
        grpc_port=6334, 
        prefer_grpc=True
        )


    db = Qdrant(client=client,
                collection_name=collection,
                embeddings=embeddings, 
                distance_strategy=distance_strategy)
    return db

def aconnect_db(collection:str, 
               distance_strategy:str = "COSINE",
               embeddings = load_embeddings()) -> Qdrant:
    """ Connect to Qdrant collection

    Args:
        collection (str): Name of collection
        distance_strategy (str, optional): Distance Estrategy #EUCLID, #COSINE, #DOT. Defaults to `COSINE`.
        embeddings (Any, optional): Embeddings model. Defaults to `load_embeddings()`.
        
    Returns:
        Qdrant: Qdrant Instance
    """    
    client = qdrant_client.QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_HOST_PORT,
        grpc_port=6334, 
        prefer_grpc=True
        )

    db = Qdrant(client=client,
                collection_name=collection,
                embeddings=embeddings, 
                distance_strategy=distance_strategy)
    return db

def save_feedback(feedback:dict, 
                  path:str = config.FEEDBACK_PATH
                  ) -> None:
    """ Save feedback

    Args:
        feedback (dict): Feedback
        path (str, optional): Path to save feedback. Defaults to `config.FEEDBACK_PATH`.
    """
    report = str(feedback)
    with open(f"{config.FEEDBACK_PATH}/feedback.txt", "a") as file:
            file.write(str(report + "\n"))
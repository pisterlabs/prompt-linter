import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from env_vars import AZURE_OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_TYPE, OPENAI_API_VERSION

def initialize_embedding_model(args):
    headers = {
        "x-api-key": AZURE_OPENAI_API_KEY,
    }
    encoder = OpenAIEmbeddings(
        deployment=args.embedding_model_id, 
        headers=headers, 
        chunk_size=1, 
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        openai_api_type=OPENAI_API_TYPE,
        openai_api_version=OPENAI_API_VERSION
    )

    return encoder

def generate_corpus_embeddings(args: object, dataloader) -> np.ndarray:
    """
        Generates embeddings for the corpus of questions in the dataset
        Args:
            args: arguments passed to the program
        Returns:
            embeddings: embeddings for the corpus of questions in the dataset
    """
    
    corpus = [example['question'] for example in dataloader]
    encoder = initialize_embedding_model(args)
    embeddings = np.array(encoder.embed_documents(corpus))
    return embeddings
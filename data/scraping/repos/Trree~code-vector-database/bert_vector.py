from langchain.embeddings import HuggingFaceEmbeddings

from src.codevecdb.config.Config import Config


def semantics_vector(code):
    cfg = Config()
    huggingface_model = cfg.vector_embeddings
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model)
    print("this is my code: " + code)
    return embeddings.embed_query(code)



from langchain.embeddings import HuggingFaceEmbeddings


def init_embedding_model():
    embed = HuggingFaceEmbeddings()
    return embed

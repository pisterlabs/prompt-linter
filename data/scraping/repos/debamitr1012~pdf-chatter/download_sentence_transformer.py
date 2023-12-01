from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
    cache_folder="./sentence_transformers"
    )
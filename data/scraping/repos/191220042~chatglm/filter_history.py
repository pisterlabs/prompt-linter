from langchain.embeddings import HuggingFaceEmbeddings
from scipy.spatial.distance import cosine

hf = HuggingFaceEmbeddings(model_name="rainjay/sbert_nlp_corom_sentence-embedding_chinese-base-ecom")
def handle_history(history:list,query:str):

    if(len(history)==0):
        return history
    new_history = [list(i) for i in history]
    embedding_history = [hf.embed_documents(i) for i in new_history]
    embedding_query = hf.embed_query(query)
    # 计算相似度
    similarity = [cosine(embedding_query, i[0]) for i in embedding_history]
    new_history = [i for i,score in zip(new_history,similarity) if score>0.7]

    history = [tuple(i) for i in new_history]

    return history


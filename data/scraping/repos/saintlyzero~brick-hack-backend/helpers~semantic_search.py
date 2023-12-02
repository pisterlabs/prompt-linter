import cohere
import numpy as np
import pinecone
from data import OS_CN, OOP
from keys import PINE_KEY, API_KEY


def init():
    co = cohere.Client(API_KEY)
    pinecone.init(
        PINE_KEY,
        environment="us-east1-gcp"  # find next to API key in console
    )
    return co


def create_index(co, index_name):
    embeds = co.embed(
        texts=OS_CN.split('.'),
        model='large',
        truncate='None'
    ).embeddings
    shape = np.array(embeds).shape
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    pinecone.create_index(
        index_name,
        dimension=shape[1],
        metric='cosine'
    )
    return embeds, shape


def upsert_data(embeds, shape, index_name):
    index = pinecone.Index(index_name)
    batch_size = 128
    ids = [str(i) for i in range(shape[0])]
    # create list of metadata dictionaries
    meta = [{'text': text} for text in OS_CN.split('.')]
    # create list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeds, meta))
    for i in range(0, shape[0], batch_size):
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])

    return index


def main():
    co = init()
    index = create_and_store(co)
    query_pinecone(co, index)


def query_pinecone(co, index):
    query1 = "Where there any announcements in the lecture?"
    # query2 = "When are the office hours?"
    # query3 = "What is an OS?"
    # query4 = "What are concepts?"
    # create the query embedding
    xq = co.embed(
        texts=[query1],
        model='large',
        truncate='None'
    ).embeddings
    res = index.query(xq, top_k=2, include_metadata=True)
    print(res)


def create_and_store(co):
    index_name = 'cohere-pinecone-os-cn'
    embeds, shape = create_index(co, index_name)
    return upsert_data(embeds, shape, index_name)


if __name__ == '__main__':
    main()

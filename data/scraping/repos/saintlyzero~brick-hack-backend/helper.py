import cohere
import pinecone
from . import django_auth
import numpy as np

co = cohere.Client(django_auth.API_KEY)
pinecone.init(
    django_auth.PINE_KEY,
    environment="us-east1-gcp"
)


def generate_summary(text):
    return co.summarize(model='summarize-xlarge', text=text, length='long', extractiveness='medium', temperature=0.25)


def generate_outline(text):
    response = co.generate(
        model='command-xlarge-20221108',
        prompt=f'extract all concepts from lecture: {text}',
        max_tokens=200,
        temperature=0,
        k=0,
        p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    return response.generations[0].text


def create_index(co, index_name, text):
    embeds = co.embed(
        texts=text.split('.'),
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


def upsert_data(embeds, shape, index_name, text):
    index = pinecone.Index(index_name)
    batch_size = 128
    ids = [str(i) for i in range(shape[0])]
    meta = [{'text': t} for t in text.split('.')]
    to_upsert = list(zip(ids, embeds, meta))
    for i in range(0, shape[0], batch_size):
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])

    return index


def query_pinecone(co, index, k):
    query1 = "Where there any announcements in the lecture?"
    xq = co.embed(
        texts=[query1],
        model='large',
        truncate='None'
    ).embeddings
    return index.query(xq, top_k=k, include_metadata=True)


def generate_announcements(text, k):
    index_name = 'cohere-pinecone-os-cn'
    # embeds, shape = create_index(co, index_name, text)
    # index = upsert_data(embeds, shape, index_name, text)
    index = pinecone.Index(index_name)
    return query_pinecone(co, index, k)


def generate_quiz(text):
    return co.generate(model='command-xlarge-20221108', prompt=f'Generate a list of 5 interview questions on {text}',
                       max_tokens=500, temperature=0, k=0, p=1, frequency_penalty=0, presence_penalty=0,
                       stop_sequences=[], return_likelihoods='NONE').generations[0].text


def remove_empty_strings(string_list):
    if type(string_list) != list:
        string_list = string_list.split('\n')
    return [string.strip() for string in string_list if string]


def semantic_search(query, k):
    xq = co.embed(
        texts=[query],
        model='large',
        truncate='None'
    ).embeddings
    index = pinecone.Index('cohere-pinecone-os-cn')
    res = index.query(xq, top_k=k, include_metadata=True)
    print(res)

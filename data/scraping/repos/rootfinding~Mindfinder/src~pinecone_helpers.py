import os
import json
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from uuid import uuid4

with open('.creds') as f:
    creds = json.load(f)
    PINECONE_API_KEY = creds['PINECONE_API_KEY']
    PINECONE_ENVIRONMENT = creds['PINECONE_ENVIRONMENT']
    OPENAI_API_KEY = creds['OPENAI_API_KEY']

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

tokenizer = tiktoken.get_encoding('cl100k_base')


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)


index_name = 'linkedin'


def query(query_text, query_filter, namespace, top_k):
    index = pinecone.GRPCIndex(index_name)
    query_vector = embed.embed_documents([query_text])[0]
    results = index.query(vector=query_vector, top_k=top_k, namespace=namespace, filter=query_filter,
                          include_metadata=True)
    return results.to_dict()


def upload(text, metadata, namespace):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )
    index = pinecone.GRPCIndex(index_name)

    uuid = str(uuid4())
    embeds = embed.embed_documents([text], 1)

    index.upsert([(uuid, embeds[0], metadata)], namespace=namespace)

    return [uuid]


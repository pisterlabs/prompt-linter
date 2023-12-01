import os

import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

load_dotenv(find_dotenv(), override=True)

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
def text_splitting():
    with open('files/churchil_speech.txt') as f:
        churchill_speech = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.create_documents([churchill_speech])
    # print(chunks[2])
    # print(chunks[10].page_content)
    # print(len(chunks))
    return chunks


def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total tokens: {total_tokens}')
    print(f'Embedding cost in USD: {total_tokens/1000*0.004:.6f}')


# def embedding_values(text):
embedding = OpenAIEmbeddings()
# vector = embedding.embed_query(text[0].page_content)
# print(vector)

def delete_all_indexes():
    print("Starting deleting indexes")
    indexes = pinecone.list_indexes()
    for id in indexes:
        pinecone.delete_index(id)
    print("Done deleting indexes")

def create_index(index_name):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        print("Done creating index")
    else:
        print("Index already exists")
chunks = text_splitting()
# print_embedding_cost(chunks)
# embedding_values(chunks)
# create_index('churchill-speech')
vector_store = Pinecone.from_documents(chunks, embedding, index_name='churchill-speech')
# B. E. F. once again under its gallant Commander in Chief,
query = "Who was the gallant commander in Chief of B. E. F?"
result = vector_store.similarity_search(query)
print(result)

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
answer = chain.run(query)
print(answer)
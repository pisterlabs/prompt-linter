import os
import sys
import time
import uuid

import openai
import openai.embeddings_utils
import pinecone
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='gcp-starter')


pinecone_index_name = 'document-search-index'


def load_documents():
    documents = []
    documents_path = 'data'
    for filename in os.listdir(documents_path):
        file_path = os.path.join(documents_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        documents.append({'title': filename.split('.')[0], 'content': content})
    return documents


def load_document_content(title):
    documents_path = 'data'
    file_path = os.path.join(documents_path, title + '.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def create_pinecone_index():
    pinecone.create_index(pinecone_index_name, metric='cosine', dimension=1536)


def fill_pinecone_index(documents):
    index = pinecone.Index(pinecone_index_name)
    for doc in documents:
        try:
            embedding_vector = get_embedding_vector_from_openai(doc['content'])
            data = pinecone.Vector(
                id=str(uuid.uuid4()),
                values=embedding_vector,
                metadata={'title': doc['title']}
            )
            index.upsert([data])
            print(f'Embedded and inserted document with title ' + doc['title'])
            time.sleep(1)
        except:
            print(f'Could not embed and insert document with title ' + doc['title'])


def query_pinecone_index(query):
    index = pinecone.Index(pinecone_index_name)
    query_embedding_vector = get_embedding_vector_from_openai(query)
    response = index.query(
        vector=query_embedding_vector,
        top_k=1,
        include_metadata=True
    )
    return response['matches'][0]['metadata']['title']


def get_embedding_vector_from_openai(text):
    return openai.embeddings_utils.get_embedding(text, engine='text-embedding-ada-002')


def get_answer_from_openai(question):
    relevant_document_title = query_pinecone_index(question)
    print(f'Relevant document title: {relevant_document_title}')
    document_content = load_document_content(relevant_document_title)
    prompt = create_prompt(question, document_content)
    print(f'Prompt:\n\n{prompt}\n\n')
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    return completion.choices[0].message.content


def create_prompt(question, document_content):
    return 'You are given a document and a question. Your task is to answer the question based on the document.\n\n' \
           'Document:\n\n' \
           f'{document_content}\n\n' \
           f'Question: {question}'


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == 'create_index':
        create_pinecone_index()
    elif arg == 'fill_index':
        documents = load_documents()
        fill_pinecone_index(documents)
    elif arg == 'get_answer':
        question = input('Enter a question: ')
        answer = get_answer_from_openai(question)
        print(answer)
    else:
        print('Invalid argument')

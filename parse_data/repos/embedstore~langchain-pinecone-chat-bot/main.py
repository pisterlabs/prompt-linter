import os
import openai
import pinecone
import itertools

from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for, session
from embedstore import load_embedding

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_DIMENSION = 1536
INDEX_NAME = "naval-almanack-book"

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))


def get_embedding(chunk):
    """Get embedding using OpenAI"""
    response = openai.Embedding.create(
        input=chunk,
        model="text-embedding-ada-002",
    )
    embedding = response['data'][0]['embedding']
    return embedding


def get_response_from_openai(query, documents):
    """Get ChatGPT api response"""
    prompt = get_prompt_for_query(query, documents)
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0,
        max_tokens=800,
        top_p=1,
    )
    return response["choices"][0]["message"]["content"]


def create_pinecone_index(index_name):
    """Create Pinecone index if it doesn't exists"""
    existing_indexes = pinecone.list_indexes()
    if index_name not in existing_indexes:
        print(f"{index_name} index not found in pinecone. Creating it...")
        pinecone.create_index(index_name, dimension=EMBEDDING_DIMENSION)
    return pinecone.Index(index_name)


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def get_prompt_for_query(query, documents):
    """Build prompt for question answering"""
    template = """
    You are given a paragraph and a query. You need to answer the query on the basis of paragraph. If the answer is not contained within the text below, say \"Sorry, I don't know. Please try again.\"\n\nP:{documents}\nQ: {query}\nA:
    """
    final_prompt = template.format(
        documents=documents,
        query=query
    )
    return final_prompt


def search_for_query(query):
    """Main function to search answer for query"""
    output = {}
    query_embedding = get_embedding(query)
    print(f"Embedding generated for {query}")
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_values=False,
        include_metadata=True,
    )
    documents = [
        match['metadata']['document'] for match in results['matches']
    ]
    documents_as_str = "\n".join(documents)
    response = get_response_from_openai(query, documents_as_str)
    print(f"Final response received from openai.")
    output["response"] = response
    output["documents"] = documents
    return output


index = create_pinecone_index(INDEX_NAME)
result = load_embedding(os.getenv("EMBEDDING_ID"), embed_for="chroma")
doc_ids = result["ids"]
embeddings = result["embeddings"]
documents = result["documents"]

final_data = []
for idx, doc_id in enumerate(doc_ids):
    final_data.append((doc_id, embeddings[idx], {"document": documents[idx]}))

for ids_vectors_chunk in chunks(final_data, batch_size=50):
    index.upsert(vectors=ids_vectors_chunk)

print(pinecone.describe_index(INDEX_NAME))


@app.route('/', methods=['GET', 'POST'])
def run_bot():
    if request.method == 'POST':
        query = request.form.get('query')
        result = search_for_query(query)
        session['result'] = result
        session['query'] = query
        return redirect(url_for('run_bot'))
    return render_template('index.html', query=session.get("query"), result=session.get("result"))


if __name__ == '__main__':
   app.run()
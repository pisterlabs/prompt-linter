import os
import functions_framework

from langchain.llms import OpenAI 
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone 

openai_key = os.environ.get('OPENAI_API_KEY')
pinecone_key = os.environ.get('PINECONE_API_KEY')
pinecone_environment = os.environ.get('PINECONE_ENVIRONMENT')
pinecone_index = "langchain1"

def answer_question(question: str, vs, chain):
    relevant_docs = vs.similarity_search(question)
    answer = chain.run(input_documents=relevant_docs, question=question)
    docs_metadata = []
    for doc in relevant_docs:
        metadata = doc.metadata
        if metadata is not None:
            doc_metadata = {
                "title": metadata.get('title', None),
                "relURI": metadata.get('relURI', None)
            }
            docs_metadata.append(doc_metadata)

    return {"answer": answer, "docs": docs_metadata}

def convert_to_document(message):
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    return Document(page_content=message, metadata={})

llm = OpenAI(temperature=1, openai_api_key=openai_key, max_tokens=-1, streaming=True) 
chain = load_qa_chain(llm, chain_type="stuff")
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
docsearch = Pinecone.from_existing_index(pinecone_index, embeddings)

import functions_framework

@functions_framework.http
def start(request):
    # For more information about CORS and CORS preflight requests, see:
    # https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'query' in request_json:
        question = request_json['query']

    elif request_args and 'query' in request_args:
        question = request_args['query']
    else:
        question = 'What is Pachyderm?'


    return (answer_question(question=question, vs=docsearch, chain=chain), 200, headers)
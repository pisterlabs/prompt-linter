from flask import Flask, request, jsonify
import weaviate
from langchain.embeddings import CohereEmbeddings, HuggingFaceBgeEmbeddings
import cohere


app = Flask(__name__)

COHERE_API_KEY = 'XXX'
WEAVIATE_API_KEY = 'XXX'
WEAVIATE_URL = 'XXX'

# client end points
co = cohere.Client(COHERE_API_KEY)

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthClientPassword(
        username = 'YOUR_USER_NAME',  # Replace w/ your WCS username
        password = "PASSWORD",  # Replace w/ your WCS password
    ),
)


@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/api/query', methods=['POST'])
def response():
    # data acquisition
    data = request.json  # json (if content-type of application/json is sent with the request)
    query = data.get('query')

    response = co.embed(
        texts=[query],
        model='embed-english-v3.0',
        input_type='search_query'
    )

    embedding = response.embeddings[0]

    result = (
        client.query
        .get("class_name", ["text", "source", "filename", "page_number"])
        .with_near_vector({
            "vector": embedding,
            "certainty": 0.7
        })
        .with_limit(10)
        .do()
    )

    data = result['data']['Get']['Cfa_proposal']

    message = {'results':[]}

    for d in data:
        message['results'].append(
            {
                'title':d['source'],
                'snippets':d['text'],
                'page_number':d['page_number'],
                'filename':d['filename']
            }
        )

    resp = jsonify(message)
    resp.status_code = 200

    return resp
    

if __name__ == '__main__':
    app.run()
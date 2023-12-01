import os
import urllib.parse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import pinecone
import pandas as pd
import numpy as np
import cohere
from cohere.classify import Example

def get_api_key(var):
    key = os.environ.get(var)
    if key is None:
        raise ValueError("Environment variable not set")
    return key

print("loading file")
# TODO edit file path to the database
file = 'PATH TO DATABASE'
df = pd.read_csv(file)
allBooks = pd.read_csv(file)
df = df.iloc[:, 1:]
print("Loading the AI!!")
summaries = df.loc[:, "short_sums"]
sum_list = summaries.to_list()

#!pip install -U cohere pinecone-client datasets

# Replace with your API key
co = cohere.Client(get_api_key("COHEREAPI_KEY"))

embeds = co.embed(
    texts=sum_list,
    model='small',
    truncate='LEFT'
).embeddings

shape = np.array(embeds).shape

pinecone.init("15c4f0ad-5cdb-4b61-a504-c25e2008b39f",
              environment='us-west1-gcp')

index_name = 'cohere-pinecone-trec'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=shape[1],
        metric='cosine'
    )

# connect to index
index = pinecone.Index(index_name)

batch_size = 128

ids = [str(i) for i in range(shape[0])]
# create list of metadata dictionaries
meta = [{'text': text} for text in sum_list]

# create list of (id, vector, metadata) tuples to be upserted
to_upsert = list(zip(ids, embeds, meta))

for i in range(0, shape[0], batch_size):
    i_end = min(i+batch_size, shape[0])
    index.upsert(vectors=to_upsert[i:i_end])

# let's view the index statistics
# print(index.describe_index_stats())


def query_call(query):

    # Create the query embedding
    xq = co.embed(
        texts=[query],
        model='small',
        truncate='LEFT'
    ).embeddings

    print(np.array(xq).shape)

    # query, returning the top 3 most similar results
    res = index.query(xq, top_k=4, include_metadata=True)

    arr = []
    for match in res['matches']:
        arr.append(allBooks.loc[allBooks['short_sums'] ==
                   match['metadata']['text']].iloc[0].to_dict())
    return arr


print("Se")
#Python 3 server example

hostName = "localhost"
serverPort = 8080


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        title = urllib.parse.parse_qs(self.path.replace('/', ''))

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if('query' in title.keys()):
            results = query_call(title['query'][0])
            self.wfile.write(
                bytes(json.dumps(results), "utf-8"))


if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

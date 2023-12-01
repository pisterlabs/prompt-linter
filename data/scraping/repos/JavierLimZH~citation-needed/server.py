import flask
from flask import request
from flask_cors import CORS

import pickle

import cohere
import toml
from subject import SubjectExtractor

import wiki

import numpy as np

config = toml.load("config.toml")
co = cohere.Client(config['cohere']['key'])
subject_extractor = SubjectExtractor(co)

nnembeddings = pickle.load(open("nnembeddings.pickle", "rb"))
nnembeddings = np.array(nnembeddings)

nnlist = pickle.load(open("nnlist.pickle", "rb"))
nntexts = nnlist["texts"]
nncitations = nnlist["citations"]
nncitation_map = nnlist["citation_map"]
nnpages = nnlist["pages"]


def narrow_search(query):
    query = query.strip()

    query_embed = np.array(co.embed([query]).embeddings)

    scores = np.dot(nnembeddings, query_embed.T).flatten()
    scores = scores / np.linalg.norm(nnembeddings, axis=1)
    scores = scores / np.linalg.norm(query_embed)

    for idx in reversed(np.argsort(scores)):
        citations = set(nncitations[idx])
        if len(citations) == 0:
            continue
        text = nntexts[idx]
        if len(text) < 10:
            continue
        citations = [nncitation_map[citation] for citation in citations]
        page = nnpages[idx]
        break

    return {
        "text": text,
        "citations": citations,
        "page": page,
    }

def general_search(query):
    query = query.strip()
    subject = subject_extractor.query(query)
    
    pages = wiki.wikipedia.search(subject)

    merged_texts = []
    merged_citations = []
    merge_citation_map = {}
    merged_pages = []

    for page in pages[:3]:
        soup = wiki.fetch_page(page)
        pairs, citations = wiki.extract_text_with_citations(soup)

        merged_texts += [pair[0] for pair in pairs]
        merged_citations += [[page+x for x in pair[1]] for pair in pairs]
        merge_citation_map.update({page+x: citations[x] for x in citations})
        merged_pages += [page] * len(pairs)

    embeddings = np.array(co.embed(merged_texts).embeddings)
    query_embed = np.array(co.embed([query]).embeddings)
    
    scores = np.dot(embeddings, query_embed.T).flatten()
    scores = scores / np.linalg.norm(embeddings, axis=1)
    scores = scores / np.linalg.norm(query_embed)

    for idx in reversed(np.argsort(scores)):
        citations = set(merged_citations[idx])
        if len(citations) == 0:
            continue
        text = merged_texts[idx]
        if len(text) < 10:
            continue
        citations = [merge_citation_map[citation] for citation in citations]
        page = merged_pages[idx]
        break

    return {
        "text": text,
        "citations": citations,
        "page": page
    }

app = flask.Flask(__name__)
CORS(app)

# serve from /dist
@app.route("/")
def index():
    return flask.send_from_directory("dist", "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return flask.send_from_directory("dist", path) 

# general search
@app.route("/general/<query>")
def general(query):
    # run 
    result = general_search(query)

    print(result)

    # return json
    return flask.jsonify(result)

@app.route("/narrow/<query>")
def narrow(query):
    # run 
    result = narrow_search(query)

    print(result)

    # return json
    return flask.jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

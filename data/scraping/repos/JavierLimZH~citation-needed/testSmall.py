import pickle
import toml 

import numpy as np

import cohere

nnembeddings = pickle.load(open("nnembeddings.pickle", "rb"))
nnembeddings = np.array(nnembeddings)

nnlist = pickle.load(open("nnlist.pickle", "rb"))
texts = nnlist["texts"]
citations = nnlist["citations"]
citation_map = nnlist["citation_map"]
pages = nnlist["pages"]

config = toml.load("config.toml")
co = cohere.Client(config['cohere']['key'])

claim = """
The sigmoid function can lead to the vanishing gradient problem.
""".strip()

claim_embed = np.array(co.embed([claim]).embeddings)

scores = np.dot(nnembeddings, claim_embed.T).flatten()
scores = scores / np.linalg.norm(nnembeddings, axis=1)
scores = scores / np.linalg.norm(claim_embed)

for i in np.argsort(scores)[::-1][:3]:
    print("="*40)
    print(pages[i])
    print(scores[i])
    
    print(texts[i])
    print("-"*40)

    for citation in set(citations[i]):
        print(citation_map[citation])

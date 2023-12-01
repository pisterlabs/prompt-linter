import toml
import pickle
import cohere

with open("nnlist.pickle", "rb") as f:
    data = pickle.load(f)

texts = data["texts"]

config = toml.load("config.toml")
co = cohere.Client(config['cohere']['key'])

# group into 100s
groups = [texts[i:i+100] for i in range(0, len(texts), 100)]
embeds = []

import time

for group in groups:
    print(f"Processing {len(group)} texts, {len(embeds)} total")
    embeds += co.embed(group).embeddings
    print("sleeping", end="\r")
    time.sleep(70)

# save the embeddings
with open("nnembeddings.pickle", "wb") as f:
    pickle.dump(embeds, f)
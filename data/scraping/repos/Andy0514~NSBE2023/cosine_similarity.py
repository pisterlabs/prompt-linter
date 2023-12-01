import numpy as np
import pickle
import cohere
co = cohere.Client('yiOWD4KfXSiayGiim2MRmZRUvGsbdEFOY5QaCQ1Z') # This is your trial API key


def load_embedding():

    # Read dictionary pkl file
    with open('embeddings.pkl', 'rb') as fp:
        embed = pickle.load(fp)

    return embed

embed = load_embedding()

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def find_closest_word(word):
    closest_match = ""
    largest_sim = 0
    word_embed = co.embed(texts=[word], model="small").embeddings[0]

    for k, v in embed.items():
        curr = cosine_similarity(v, word_embed)
        if (k == "absorb"):
            print(v)
            print(word_embed)
            print(curr)

        if (curr > largest_sim):
            largest_sim = curr
            closest_match = k

    return closest_match

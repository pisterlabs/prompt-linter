from sklearn.metrics.pairwise import cosine_similarity
import cohere
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
load_dotenv()
key = os.getenv('api_key')
co = cohere.Client(key)
result = os.path.dirname(os.path.realpath('__file__'))
relative_path = ".\\data\\embeddings.npy"
full_path = os.path.join(result, relative_path)


def calculate_embeddings():
    df = pd.read_csv(
        "C:\\Users\\bouta\\OneDrive\\Bureau\\hackathons\\nextGenAI\\data\\data.csv")
    embeds = co.embed(texts=list(
        df['job_description']), model='embed-english-v2.0').embeddings
    all_embeddings = np.array(embeds)

    np.save(full_path, all_embeddings)


def embed_text(texts):
    embeddings = co.embed(
        model='embed-english-v2.0',
        texts=texts)
    embeddings = np.array(embeddings.embeddings)
    return embeddings


def get_similarity(target):

    cosine_similarity_list = []
    all_embeddings = np.load(
        "C:\\Users\\bouta\\OneDrive\\Bureau\\hackathons\\nextGenAI\\data\\embeddings.npy")
    for i in range(len(all_embeddings)):
        # Calculate the dot product of the two arrays
        dot_product = np.dot(target, all_embeddings[i])

        # Calculate the norm of the two arrays
        norm_embedding_for_recipe_test = np.linalg.norm(target)
        norm_embedding = np.linalg.norm(all_embeddings[i])

        # Calculate the cosine similarity
        cosine_similarity = dot_product / \
            (norm_embedding_for_recipe_test * norm_embedding)

        cosine_similarity_list.append((i, np.array((cosine_similarity))[0]))

    # Sort the list
    cosine_similarity_list.sort(key=lambda x: x[1], reverse=True)

    # Select the top 3 by ID
    top_3 = [x for x in cosine_similarity_list[:3]]

    return top_3


def main():
    # calculate_embeddings()
    target = ["Computer Science Engineering,French,English,Windows,Linux,Oracle,MySQL,PostgreSQL,MongoDB,SQLITE3,Python,Java,Javascript,c/c++,php,HTML,CSS,JS,React,Vuejs,Django,Flask,Node js,Expressjs,Laravel,Docker,Kubernetes,Git,VRP,IOT,biometric,attendency detection,web-sockets,arduino,M2M Services,El Jadida,Python,html,css,javascript,Kubernetes,Docker,CI/CD,Tekton CI/CD,Laravel,Vue.js,functional requirements"]
    print(get_similarity(embed_text(target)))


if __name__ == "__main__":
    main()

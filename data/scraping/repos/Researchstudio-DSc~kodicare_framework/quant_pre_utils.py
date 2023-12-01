import json
import os
import pickle
import pandas as pd
import argparse
import re
import nltk
from nltk.corpus import stopwords
import umap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import optuna
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from typing import List, Iterator
import random

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



def path_exits(path):
    return os.path.exists(path)


def mkdir(path):
    if not path_exits(path):
        os.mkdir(path)


def list_files_in_dir(dir):
    return [file for file in os.listdir(dir) if is_file(join(dir, file))]


def list_directories(dir):
    return [subdir for subdir in os.listdir(dir) if os.path.isdir(join(dir, subdir))]


def is_file(path):
    return os.path.isfile(path)


def join(path1, path2):
    return os.path.join(path1, path2)


def write_json(path, dict):
    with open(path, 'w') as outfile:
        json.dump(dict, outfile, indent=2)
    outfile.close()


# def read_json(path):
#     with open(path, "r") as infile:
#         data = json.load(infile)
#     infile.close()
#     return data

def read_json(path):
    """Pandas read_json returns the collection in a list
    of elements with the number, id and contents"""
    data = pd.read_json(path)
    return data


def read_json_v2(path):
    """Pandas read_json returns the collection in a list
    of elements with the number, id and contents"""
    data = pd.read_json(path)
    list_content = data.contents.tolist()
    return list_content


def read_file_into_list(input_file):
    lines = []
    with open(input_file, "r") as infile_fp:
        for line in infile_fp.readlines():
            lines.append(line.strip())
    infile_fp.close()
    return lines


def write_list_to_file(output_file, list):
    with open(output_file, "w") as outfile_fp:
        for line in list:
            outfile_fp.write(line + "\r\n")
    outfile_fp.close()


def write_text_to_file(output_file, text):
    with open(output_file, "w") as output_fp:
        output_fp.write(text)
    output_fp.close()


def write_pickle(data, file_path):
    pickle.dump(data, open(file_path, "wb"))


def read_pickle(file_path):
    return pickle.load(open(file_path, 'rb'))


def preprocess_and_clean_text_alpha(text):
    """ This function will remove special characters, convert text to lowercase, and remove stopwords """

    stop_words = set(stopwords.words('english'))

    # Remove special characters and numbers using regular expressions
    cleaned_text = re.sub(r'\W+|\d+', ' ', text)

    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(cleaned_text)

    # Remove stopwords and join tokens back into a single string
    cleaned_text = ' '.join(token for token in tokens if token not in stop_words)

    return cleaned_text


def preprocess_and_clean_text(text):
    """ This function will remove special characters, handle contractions, convert text to lowercase,
    remove stopwords, and lemmatize the tokens """

    # Set up a more comprehensive list of stopwords
    stop_words = set(stopwords.words('english'))

    # Set up the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove special characters and numbers using regular expressions
    cleaned_text = re.sub(r'\W+|\d+', ' ', text)

    # Handle contractions
    cleaned_text = contractions.fix(cleaned_text)

    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()

    # Tokenize the text
    tokens = word_tokenize(cleaned_text)

    # Remove stopwords and lemmatize the tokens
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    # Join the cleaned tokens back into a single string
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text


def preprocess_and_clean_text_v2(text):
    """ This function will remove special characters, handle contractions, convert text to lowercase,
    remove stopwords, and lemmatize the tokens """

    # Set up a more comprehensive list of stopwords
    stop_words = set(stopwords.words('english'))

    # Set up the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove special characters and numbers using regular expressions
    cleaned_text = re.sub(r'\W+|\d+', ' ', text)

    # Handle contractions
    cleaned_text = contractions.fix(cleaned_text)

    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()

    # Tokenize the text
    tokens = word_tokenize(cleaned_text)

    # Remove stopwords and lemmatize the tokens
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    # Join the cleaned tokens back into a single string
    cleaned_text = ' '.join(cleaned_tokens)

    yield cleaned_text


def objective(trial, doc_embeddings):

    """The function create a simple scoring function based on the pairwise distances between
    the high-dimensional and low-dimensional embeddings to set automatically the parameters for the umap_vis function.

    It uses the euclidean_distances function from sklearn.metrics.pairwise to compute distances between embeddings
    in the high-dimensional and low-dimensional spaces.
    The objective will be to minimize the difference between these distances.

    Parameters
    ----------
    trial : leave it by default, do not touch it
    doc_embeddings : embedded document vectors from the top2vec model

    Returns
    -------
    score
        to be given to the umap function
    """

    n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
    min_dist = trial.suggest_float("min_dist", 0.0, 1.0)
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        random_state=42,
    )
    umap_embeddings = umap_model.fit_transform(doc_embeddings)

    # Scoring based on pairwise distance difference
    high_dim_distances = euclidean_distances(doc_embeddings)
    low_dim_distances = euclidean_distances(umap_embeddings)

    score = -np.mean(np.abs(high_dim_distances - low_dim_distances))
    return score


def umap_vis(model, num_parent_topics, optimization=False):

    """To visualize the results from Top2Vec, this function uses UMAP, following these steps:
    - Extract the document embeddings from the Top2Vec model.
    - Use UMAP for dimensionality reduction to 2D space.
    - Visualize the reduced document embeddings using a scatter plot.

    Parameters
    ----------
    model : The top2vec model
    optimization : set to True/False if you want to let the optima algorithm to set the parameters of Umap, or
                   use the default ones

    Returns
    -------
    png
        2D plot with the umap result and the topics distribution highlighted in different colors
    """

    # Perform hierarchical topic reduction
    num_parent_topics = num_parent_topics
    model.hierarchical_topic_reduction(num_parent_topics)
    doc_embeddings = model.document_vectors

    if optimization is False:

        umap_embeddings = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, random_state=42).fit_transform(
            doc_embeddings)

    else:

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, doc_embeddings), n_trials=50)
        best_params = study.best_params
        print("Best parameters: ", best_params)

        umap_model = umap.UMAP(
            n_neighbors=best_params["n_neighbors"],
            n_components=2,
            min_dist=best_params["min_dist"],
            random_state=42,
        )
        umap_embeddings = umap_model.fit_transform(doc_embeddings)

    # Create a list of document IDs based on the length of the document vectors
    doc_ids = list(range(len(model.document_vectors)))

    # Get topic assignments for each document
    topic_assignments = model.get_documents_topics(doc_ids, reduced=True)[0]

    # Create a color map for topics
    num_topics = len(np.unique(topic_assignments))
    colors = plt.get_cmap('rainbow')(np.linspace(0, 1, num_topics))

    plt.figure(figsize=(10, 8))

    # Plot the dots with colors corresponding to their topics
    for topic_id, color in enumerate(colors):
        topic_docs = umap_embeddings[topic_assignments == topic_id]
        plt.scatter(topic_docs[:, 0], topic_docs[:, 1], s=10, color=color, edgecolors='none', alpha=0.8,
                    label=f'Topic {topic_id}')

    plt.title('UMAP projection of document embeddings', fontsize=16)
    plt.xlabel('UMAP 1', fontsize=14)
    plt.ylabel('UMAP 2', fontsize=14)
    plt.legend()
    plt.show()

    # Save the UMAP visualization
    plt.savefig("top2vec_umap_visualization.png", dpi=300)


def compute_umass_coherence(model, num_parent_topics, corpus, dictionary):
    """Function to compute UMass coherence for a given number of parent topics"""

    model.hierarchical_topic_reduction(num_parent_topics)
    reduced_topic_words, _, _ = model.get_topics(reduced=True)
    cm = CoherenceModel(topics=reduced_topic_words, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    return cm.get_coherence()


def parent_topics(model, max_num_topics):

    # Get the document vectors and topic words
    topic_words, _, _ = model.get_topics()

    # Tokenize the documents and create a dictionary and corpus for gensim coherence model
    texts = model.documents
    tokenized_texts = [nltk.word_tokenize(doc) for doc in texts]
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # Try different numbers of parent topics and compute their coherence
    min_parent_topics = np.int(max_num_topics/3)
    max_parent_topics = (max_num_topics - 1)
    parent_topics_range = range(min_parent_topics, max_parent_topics + 1)
    coherence_scores = []

    for num_parent_topics in parent_topics_range:
        coherence_score = compute_umass_coherence(model, num_parent_topics, corpus, dictionary)
        coherence_scores.append(coherence_score)

    # Plot the coherence scores against the number of parent topics
    plt.plot(parent_topics_range, coherence_scores, marker='o')
    plt.xlabel("Number of Parent Topics")
    plt.ylabel("UMass Coherence")
    plt.title("Elbow Method for Optimal Number of Parent Topics")
    plt.xticks(np.arange(min_parent_topics, max_parent_topics + 1, step=1))
    plt.show()

    # Find the optimal number of parent topics using the elbow method
    optimal_parent_topics = parent_topics_range[np.argmax(np.gradient(coherence_scores))]
    print(f"Optimal number of parent topics: {optimal_parent_topics}")

    return optimal_parent_topics


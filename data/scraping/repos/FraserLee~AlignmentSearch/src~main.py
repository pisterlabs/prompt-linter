import openai
"""
import config
from assistant.semantic_search import AlignmentSearch
from dataset.create_dataset import Dataset

openai.api_key = config.OPENAI_API_KEY

from settings import PATH_TO_RAW_DATA, PATH_TO_DATASET, EMBEDDING_MODEL, LEN_EMBEDDINGS
"""
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import numpy as np

import sys
import pickle
from pathlib import Path
import random

src_path = Path(__file__).resolve().parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from dataset import create_dataset
#from assistant import semantic_search
from settings import EMBEDDING_MODEL, PATH_TO_DATASET_DICT_PKL

import numpy as np
import matplotlib.pyplot as plt


def load_rawdata_into_pkl():
    """with open(PATH_TO_DATASET, 'rb') as f:
    dataset = pickle.load(f)
    AS = AlignmentSearch(dataset=dataset)
    prompt = "What would be an idea to solve the Alignment Problem? Name the Lesswrong post by Quintin Pope that discusses this idea."
    answer = AS.search_and_answer(prompt, 3, HyDE=False)
    print(answer)
    """  
    # List of possible sources:
    all_sources = ["https://aipulse.org", "ebook", "https://qualiacomputing.com", "alignment forum", "lesswrong", "manual", "arxiv", "https://deepmindsafetyresearch.medium.com", "waitbutwhy.com", "GitHub", "https://aiimpacts.org", "arbital.com", "carado.moe", "nonarxiv_papers", "https://vkrakovna.wordpress.com", "https://jsteinhardt.wordpress.com", "audio-transcripts", "https://intelligence.org", "youtube", "reports", "https://aisafety.camp", "curriculum", "https://www.yudkowsky.net", "distill", 
                   "Cold Takes", "printouts", "gwern.net", "generative.ink", "greaterwrong.com"] # These last do not have a source field in the .jsonl file

    # List of sources we are using for the test run:
    custom_sources = [
        "https://aipulse.org", 
        "ebook", 
        "https://qualiacomputing.com", 
        "alignment forum", 
        "lesswrong", 
        "manual", 
        "arxiv", 
        "https://deepmindsafetyresearch.medium.com/", 
        "waitbutwhy.com", 
        "GitHub", 
        "https://aiimpacts.org", 
        "arbital.com", 
        "carado.moe", 
        "nonarxiv_papers", 
        "https://vkrakovna.wordpress.com", 
        "https://jsteinhardt.wordpress.com", 
        "audio-transcripts", 
        "https://intelligence.org", 
        "youtube", 
        "reports", 
        "https://aisafety.camp", 
        "curriculum", 
        "https://www.yudkowsky.net", 
        "distill",
        "Cold Takes",
        "printouts",
        "gwern.net",
        "generative.ink",
        "greaterwrong.com"
    ]
    
    dataset = create_dataset.Dataset(
        custom_sources=custom_sources, 
        rate_limit_per_minute=3500, 
        min_tokens_per_block=200, max_tokens_per_block=300, 
        # fraction_of_articles_to_use=1/150,
    )
    dataset.get_alignment_texts()
    
    print(len(dataset.embedding_strings))
    print(dataset.total_word_count)
    print(dataset.total_block_count)
    print(dataset.articles_count)
    
    dataset.get_embeddings()
    dataset.save_data()
    
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
def get_embedding(text: str) -> np.ndarray:
    result = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
    return np.array(result["data"][0]["embedding"])

def print_out_dataset_stuff():
    with open(PATH_TO_DATASET_PKL, 'rb') as f:
        dataset = pickle.load(f)

    embeddings_len = len(dataset.embedding_strings)
    i1 = random.randint(0,embeddings_len-1)
    #i2 = random.randint(0,embeddings_len-1)
    #print(len(dataset.embeddings))
    #print(len(dataset.embedding_strings))
    #embedding_test = get_embedding(dataset.embedding_strings[i])
    #print(np.dot(embedding_test,dataset.embeddings[i]))

    metadata_i1 = dataset.embeddings_metadata_index[i1]
    print("metadata:",dataset.metadata[metadata_i1])
    print("embedding_string:",dataset.embedding_strings[i1])
    #print("embedding_vector:",dataset.embeddings[i1])
    embedding_of_string1 = get_embedding(dataset.embedding_strings[i1])

    #metadata_i2 = dataset.embeddings_metadata_index[i2]
    #print("metadata:",dataset.metadata[metadata_i2])
    #print("embedding_string:",dataset.embedding_strings[i2])
    #print("embedding_vector:",dataset.embeddings[i1])
    #embedding_of_string2 = get_embedding(dataset.embedding_strings[i2]) 
    #embedding_of_string2 = get_embedding("000000000000000000000000000000000000000000000000000000000000000000000000000000000")

    #print(len(embedding_of_string1))
    vector = dataset.embeddings[i1]
    plot_likelihood(vector)
    #plot_likelihood(get_embedding("tst"))
    print(max(vector), min(vector))
    print(sum([x**2 for x in vector]))


    

    #print(np.dot(embedding_of_string1, embedding_of_string2))

def plot_likelihood(embeddings, num_buckets=200):
    # Calculate the histogram
    histogram, bin_edges = np.histogram(embeddings.flatten(), bins=num_buckets, range=(embeddings.min(), embeddings.max()))

    # Normalize the histogram to get likelihoods
    likelihoods = histogram / embeddings.flatten().size

    # Plot the likelihoods
    plt.bar(bin_edges[:-1], likelihoods, width=(bin_edges[1] - bin_edges[0]), edgecolor="k", alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Likelihood")
    plt.title("Likelihood of Floats in the Vector Embedding")
    plt.savefig("bla.png")

    
    



if __name__ == "__main__":
    # load_rawdata_into_pkl()
    # print_out_dataset_stuff()
    
    with open(PATH_TO_DATASET_DICT_PKL, 'rb') as f:
        dataset = pickle.load(f)
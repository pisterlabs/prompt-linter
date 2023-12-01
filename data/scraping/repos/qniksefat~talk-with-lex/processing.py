import numpy as np
import cohere
from annoy import AnnoyIndex
import warnings
import os
import time

RATE_LIMIT = 100  # Requests per minute
SECONDS_PER_MINUTE = 60


def load_paragraphs(directory) -> np.array:
    """ Loads all paragraphs from a directory
    Each paragraph is separated by <<<{filenumber}>>>
    But we want to keep the paragraph number too
    So, we append `<<<` at the beginning of each paragraph
    
    Returns:
        np.array: Array of all paragraphs
    """
    paragraphs = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as file:
                file_paragraphs = file.read().split('<<<')[1:]
                file_paragraphs = ['<<<' + f for f in file_paragraphs if len(f) > 500]  # Remove short paragraphs
                paragraphs.extend(file_paragraphs)
    return np.array(paragraphs)


def process_and_save_chunks(text_chunks, api_key, output_directory):
    """Processes and saves text chunks in batches of 100.
    
    How much it takes? each episode is roughly 100_000 characters, 
    each chunk is 2000 characters, so 50 chunks per episode.
    For ~350 episodes, 50 * 350 = 17_500 chunks / 100 = 175 minutes = 3 hours.
    
    Args:
        text_chunks (np.array): Array of text chunks/paragraphs.
        api_key (str): API key for the Cohere API.
        output_directory (str): Directory to save the processed chunks.
    """
    if not os.path.exists(output_directory):    os.makedirs(output_directory)
    co = cohere.Client(api_key)
    
    for i in range(0, len(text_chunks), RATE_LIMIT):
        batch = text_chunks[i:i+RATE_LIMIT]
        embeddings = co.embed(texts=batch.tolist()).embeddings
        np.save(f'{output_directory}embeddings_{i}.npy', embeddings)
        time.sleep(SECONDS_PER_MINUTE)


def load_embeddings(output_directory):
    """Loads and concatenates embeddings saved in batches.
    Returns:
        np.array: Concatenated embeddings shaped (num_embeddings, embedding_dim=4096 for cohere)
    """
    embeddings = []
    for file_name in os.listdir(output_directory):
        if file_name.startswith('embeddings_') and file_name.endswith('.npy'):
            file_path = os.path.join(output_directory, file_name)
            embeddings.append(np.load(file_path))
    return np.concatenate(embeddings, axis=0)


def build_search_index(embeddings):
    search_index = AnnoyIndex(embeddings.shape[1], 'angular')
    for i in range(len(embeddings)):
        search_index.add_item(i, embeddings[i])
    search_index.build(10)
    search_index.save('search_index.ann')
    return search_index


def search_for_context(query, search_index, texts, num_nearest=3):
    """ Searches for similar viewpoints to the query

    Args:
        query (str): The question to search for
        search_index (AnnoyIndex): The search index to use
        texts (list): List of all viewpoints, i.e. paragraphs/chunks of text

    Returns:
        list: List of nearest viewpoints to the query
    """
    co = cohere.Client(api_key)
    query_embed = co.embed(texts=[query]).embeddings
    similar_item_ids = search_index.get_nns_by_vector(
        query_embed[0], num_nearest, include_distances=True)
    search_results = [texts[i] for i in similar_item_ids[0]]
    return search_results


def generate_response(question, contexts, api_key):
    co = cohere.Client(api_key)

    prompt = f"""
    Extract the answer from the following ideas.

    Consider the following ideas by different thinkers:
    {'--- Another IDEA: '.join(contexts)}

    Question: {question}
    
    Please note:
    - These are different views about the topic. Try to consider all of them with a critical eye.
    - If some words at end or beginning of IDEAs are broken, please correct them; it must be typos.
    """

    prediction = co.generate(
        prompt=prompt.strip(),
        max_tokens=150,
        model="command-nightly",
        temperature=0.75,    
        # temperature determines the randomness/creativity of the response
        num_generations=1
    )

    return prediction.generations


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    api_key = os.environ['COHERE_API_KEY']

    texts = load_paragraphs('processed_data/')
    
    process_and_save_chunks(texts, api_key, 'embeddings/')
    embeddings = load_embeddings('embeddings/')
        
    if os.path.exists('search_index.ann'):
        search_index = AnnoyIndex(embeddings.shape[1], 'angular')
        search_index.load('search_index.ann')
    else:
        search_index = build_search_index(embeddings)
    
    query = "What is the meaning of life?"
    print(f"\nQuestion: {query}")
    viewpoints = search_for_context(query, search_index, texts, num_nearest=3)
    results = generate_response(query, viewpoints, api_key)
    print(f"Answer: {results[0]}\n")
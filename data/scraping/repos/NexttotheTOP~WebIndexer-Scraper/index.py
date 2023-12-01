import json

from annoy import AnnoyIndex

import numpy as np
import pandas as pd

from helper import get_keys


def load_dataset(data_file) -> pd.DataFrame:
    df = pd.read_csv('data/' + data_file)[:1000]
    return df


def get_embeds_Cohere(text_list) -> np.ndarray:
    import cohere

    cohere_key, _ = get_keys()

    co = cohere.Client(cohere_key)
    embeds = co.embed(texts=text_list,
                      model='large',
                      truncate='left').embeddings

    embeds = np.array(embeds)
    return embeds


def get_embeds_AI21(text_list) -> np.ndarray:
    # Breaks the strings with 2000+ characters into smaller strings
    for i, s in enumerate(text_list):
        if len(s) > 2000:
            text_list.pop(i)
            for j in range(0, len(s), 2000):
                text_list.insert(i + j, s[j:j + 2000])

    # Post 200 strings at a time
    import requests

    _, ai21_key = get_keys()

    results = []



    """
    response = requests.post('https://api.ai21.com/studio/v1/experimental/embed',
                             json={'texts': text_list[0: 2]},
                             headers={'Authorization': f'Bearer {ai21_key}'})
    print(response.json()['results'])
    print("RESPONSE ^")
    results.extend(response.json()['results'])
    # Serializing json
    json_object = json.dumps(response.json(), indent=4)
    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)"""
    for i in range(0, len(text_list), 200):
        response = requests.post('https://api.ai21.com/studio/v1/experimental/embed',
                                 json={'texts': text_list[i:i + 200]},
                                 headers={'Authorization': f'Bearer {ai21_key}'})
        embeddings = list(map(lambda x: x["embedding"], response.json()['results']))
        results.extend(embeddings)

    return np.array(results)


def _build_index(df, index_filename) -> None:
    # Get embeds
    embeds = get_embeds_AI21(list(df['paragraphs']))
    # embeds = get_embeds_Cohere(list(df['paragraphs']))

    # Create the search index, pass the size of embedding
    search_index = AnnoyIndex(embeds.shape[1], 'angular')
    # Add all the vectors to the search index
    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])
    search_index.build(50)  # 10 trees
    search_index.save(index_filename)


def get_index(df: pd.DataFrame, index_filename: str) -> AnnoyIndex:
    import os

    index_dir = 'indexes'

    if not os.path.isdir(index_dir):
        os.mkdir(index_dir)

    index_path = os.path.join(index_dir, index_filename)
    if not os.path.isfile(index_path):
        _build_index(df, index_path)

    index = AnnoyIndex(768, 'angular')
    index.load(index_path)
    return index


def get_closest_paragraphs(df: pd.DataFrame, index: AnnoyIndex, query: str, n: int = 100) -> pd.DataFrame:
    query_embed = get_embeds_AI21([query])
    # query_embed = get_embeds_Cohere([query])

    # Retrieve nearest neighbors
    similar_item_ids = index.get_nns_by_vector(query_embed[0], n,
                                               include_distances=True)

    # Format and print the text and distances
    results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['paragraphs'],
                                 'links': df.iloc[similar_item_ids[0]]['link'],
                                 'distance': similar_item_ids[1]})

    return results


def test():
    df = load_dataset('Spring 3.2.csv')
    index = get_index(df, 'Spring 3.2.ann')
    while True:
        query = input("Enter a question: ")
        if query == 'x':
            break
        results = get_closest_paragraphs(df, index, query)
        print(results)


if __name__ == '__main__':
    test()

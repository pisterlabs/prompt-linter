import os

import cohere
import pandas as pd
import umap

from annoy import AnnoyIndex
import numpy as np
from dotenv import load_dotenv
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def get_key():
    load_dotenv()
    return os.getenv("COHERE_KEY")




def buildIndex(datafile: str, indexfile: str):
    df = pd.read_csv(datafile, encoding="ISO-8859-1")

    embeds = co.embed(texts=list(df['Summary']), model = 'large', truncate='right').embeddings

    embeds = np.array(embeds)

    search_index = AnnoyIndex(embeds.shape[1], 'angular')
    print(embeds.shape[1])

    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])

    search_index.build(10)
    search_index.save(indexfile)




def getClosestNeighbours():
    df = pd.read_csv('data.csv', encoding="ISO-8859-1")

    search_index = AnnoyIndex(4096, 'angular')
    search_index.load('test.ann')

    query = 'I want a paper on astro physics'

    query_embed = co.embed(texts=[query],
                        model='large',
                        truncate='right').embeddings


    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0],10,
                                                        include_distances=True)
    # Format the results
    print(similar_item_ids)
    results = pd.DataFrame(data={'title': df.iloc[similar_item_ids[0]]['Title'],
                                 'subject': df.iloc[similar_item_ids[0]]['Subject'],
                                  'summary': df.iloc[similar_item_ids[0]]['Summary'],
                                 'distance': similar_item_ids[1]})

    print(f"Query:'{query}'\nNearest neighbors:")
    print(results)





if __name__ == '__main__':
    key = get_key()
    co = cohere.Client(key)
    buildIndex()
    getClosestNeighbours()




    '''title = 'Cosmic Accelerators'
    summary = 'I discuss the scientific rationale and opportunities in the study of high energy particle accelerators away from the Earth; mostly, those outside the Solar System. I also briefly outline the features to be desired in telescopes used to probe accelerators studied by remote sensing.'
    prompt = 'This is a paper titled ' + title + '. This is the summary: ' + summary + '. The 3 themes from this summary are:'
    response = co.generate(prompt=prompt, p=0.0, temperature=0.0, max_tokens=50)
    print('Prediction: {}'.format(response.generations[0].text))'''


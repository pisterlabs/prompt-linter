import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
def get_emotion(query):
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_colwidth', None)
    # Get dataset
    dataset = load_dataset("emotion", split="train")
    # Import into a pandas dataframe, take only the first 1000 rows
    df = pd.DataFrame(dataset)[:1000]
    # Preview the data to ensure it has loaded correctly
    df.head(10)
    # Paste your API key here. Remember to not share publicly
    api_key = '7AdE0NfrjyQCJ3WSTD7iJvRvMZC5xPWHKKQZmHTW'

    # Create and retrieve a Cohere API key from dashboard.cohere.ai/welcome/register
    co = cohere.Client(api_key)
    labels = []
    for thing in dataset:
        labels.append(thing)
    #print(labels)
    # Get the embeddings
    embeds = co.embed(texts=list(df['text']),
                      model='large',
                      truncate='LEFT').embeddings
    #print(len(embeds))
    # Create the search index, pass the size of embedding
    search_index = AnnoyIndex(4096, 'angular')
    # Add all the vectors to the search index
    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])
    search_index.build(10) # 10 trees
    search_index.save('test.ann')
    # Choose an example (we'll retrieve others similar to it)
    example_id = 92
    # Retrieve nearest neighbors
    similar_item_ids = search_index.get_nns_by_item(example_id,10,
                                                    include_distances=True)
    # Format and print the text and distances
    results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'],
                                 'distance': similar_item_ids[1]}).drop(example_id)
    #print(f"Question:'{df.iloc[example_id]['text']}'\nNearest neighbors:")
    #print(results)

    #query = "Yesterday I went the gym and saw many people with good fitness but im very fat so i couldnt stand begin there so i came home while crying"
    #query = input("Enter any phrase that comes to your mind: ")
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                      model="large",
                      truncate="LEFT").embeddings

    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0],10,
                                                    include_distances=True)
    # Format the results
    results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'],
                                 'distance': similar_item_ids[1]})

    #print (similar_item_ids[1][0])
    #print(f"Input:'{query}'\nEmotion:")
    r = str(results).strip()
    i = 0
    result = ""
    while True:
        try:
            result += str(int(r[16 + i]))
            i += 1
        except:
            break;
    emotion = labels[int(result)]['label']
    SADNESS = 0
    JOY = 1
    LOVE = 2
    ANGER = 3
    FEAR = 4
    SURPRISE = 5
    if emotion == SADNESS:
        #print("You are feeling the emotion of sadness")
        return "sadness," + str(similar_item_ids[1][0])
    elif emotion == JOY:
        #print("You are feeling the emotion of joy")
        return "joy," + str(similar_item_ids[1][0])
    elif emotion == LOVE:
        #print("You are feeling the emotion of love")
        return "love," + str(similar_item_ids[1][0])
    elif emotion == ANGER:
        #print("You are feeling the emotion of anger")
        return "anger," + str(similar_item_ids[1][0])
    elif emotion == FEAR:
        #print("You are feeling the emotion of fear")
        return "fear," + str(similar_item_ids[1][0])
    elif emotion == SURPRISE:
        #print("You are feeling the emotion of surprise"
        return "surprise," + str(similar_item_ids[1][0])

    #print(labels[681])
    #print(labels[747])
#print(get_emotion("im terrified at this moment"))

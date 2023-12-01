
oxford_3000 = []
with open("words.txt", 'r') as f:
    oxford_3000 = [w.strip() for w in f.readlines()]

oxford_3000[4]

import openai
openai.api_key = "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"

#word_embeddings_map = {}
#
#model = "text-embedding-ada-002"
#n_items = len(oxford_3000)
#batch_size = 1000
#n_batches = (n_items + batch_size - 1) // batch_size
#for i in range(n_batches):
    #start,end = i * batch_size, (i + 1) * batch_size
    #inp = oxford_3000[start:end]
    #print(start,end)
    #response = openai.Embedding.create(input=inp, model=model)
    #embeddings = [i["embedding"] for i in response["data"]]
    #for word, embedding in zip(inp, embeddings):
        #word_embeddings_map[word] = embedding

import pickle
#with open('embeddings.pickle', 'ab') as f:
    #pickle.dump(word_embeddings_map, f)
with open('embeddings.pickle', 'rb') as f:
    word_embeddings_map = pickle.load(f)

import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

words = list(word_embeddings_map.keys())

#embeddings = []
#for w in words:
    #embeddings.append(word_embeddings_map[w])
#embeddings = np.array(embeddings)
embeddings = np.array(list(word_embeddings_map.values()))


axis = [["small", "large"],
        ["mean", "happy"],
        ["abstract", "tangible"]]

pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)


import plotly.graph_objects as go

def visualize_3d_highlight(words_to_highlight):

    # Default values for all words
    colors = ['blue' if word not in words_to_highlight else 'red' for word in words]
    opacities = [0.1 if word not in words_to_highlight else 1.0 for word in words]
    sizes = [5 if word not in words_to_highlight else 10 for word in words]
    texts = [word if word not in words_to_highlight else word for word in words]

    # Create the scatter plot
    scatter = go.Scatter3d(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        mode='markers',
        marker=dict(color=colors, size=sizes),
        text=texts
    )
    layout = go.Layout(height=800)  # Set height as per your preference
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

# Call the function and pass the words you want to highlight
words_to_highlight = ["life", "death", "awake", "asleep", "day", "night"] # replace with the words you want
visualize_3d_highlight(words_to_highlight)

#import json
#with open('reduced_embeddings.json', 'w') as f:
    #json.dump(reduced_embeddings.tolist(), f)
#
#with open('words.json', 'w') as f:
    #json.dump(words, f)

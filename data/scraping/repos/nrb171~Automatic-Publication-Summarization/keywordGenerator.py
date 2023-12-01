import openai
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# imports
import numpy as np
import pandas as pd
from ast import literal_eval
import tiktoken
import time

with open("key.ini", "r") as f:
        openai.api_key = f.read()

# load data
datafile_path = "/Users/nrb171/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/PSU/Didlake Group-PSU/Papers/embeddings/embeddings.csv"

df = pd.read_csv(datafile_path, header=None)
df["embedding"] = df[2].apply(literal_eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(df.embedding.values)
matrix.shape

n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
kmeans.fit(matrix)
labels = kmeans.labels_
df["Cluster"] = labels

df.groupby("Cluster").Score.mean().sort_values()

tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
vis_dims2 = tsne.fit_transform(matrix)

x = [x for x, y in vis_dims2]
y = [y for x, y in vis_dims2]

for category, color in enumerate(["purple", "green", "red", "blue"]):
    xs = np.array(x)[df.Cluster == category]
    ys = np.array(y)[df.Cluster == category]
    plt.scatter(xs, ys, color=color, alpha=0.3)

    avg_x = xs.mean()
    avg_y = ys.mean()

    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
plt.title("Clusters identified visualized in language 2d using t-SNE")
plt.savefig("clusters.png")

# Reading a review which belong to each group.
rev_per_cluster = 4

for i in range(n_clusters):
    print(f"Cluster {i} Theme:", end=" ")

    papers = np.array(df[1][df["Cluster"] == i])
    for j,paper in enumerate(papers):
        papers[j] = paper[:paper.find("###",100)]
    papers = "\n".join(papers)


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role":"user","content":"What do the following academic papers have in common?\n\nPaper summaries:\n\n"+papers+"\n\nTheme:"}],
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(response["choices"][0]["message"]["content"])
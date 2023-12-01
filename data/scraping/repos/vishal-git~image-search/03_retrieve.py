# retrieves images that match based on a similarity search with the search query
from langchain.vectorstores.cassandra import Cassandra
from langchain.embeddings import OpenAIEmbeddings
from PIL import Image
import gradio
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from utils import getCQLSession, getCQLKeyspace
from config import table_name, num_results, output_img_width, output_img_height

session = getCQLSession()
keyspace = getCQLKeyspace()

myCassandraVStore = Cassandra(
    embedding=OpenAIEmbeddings(),
    session=session,
    keyspace=keyspace,
    table_name=table_name,
)

def display_results(query):
    """Display the results (matching images) of a similarity search with the given query."""

    matches = myCassandraVStore.similarity_search_with_score(
        query=query,
        k=num_results,
    )

    found_images, scores = [], []

    for i, r in enumerate(matches):
        doc = r[0]
        score = r[1]
        image_path = doc.metadata['image_path']
        found_images.append(image_path)
        scores.append(score)
        plt.imshow(plt.imread(image_path))
    # pad the result in any case to the number of displayed widgets
    return found_images + [None] * (num_results - len(found_images))


image_search_ui = gradio.Interface(
    fn=display_results,
    inputs=gradio.components.Text(label="Search query"),
    outputs=[
        gradio.components.Image(label=f"Search result #{output_i}",
        width=output_img_width, height=output_img_height)
        for output_i in range(num_results)
    ],
    title="Image Search with CassIO & Vector Database",
)

image_search_ui.launch(share=True, debug=True)

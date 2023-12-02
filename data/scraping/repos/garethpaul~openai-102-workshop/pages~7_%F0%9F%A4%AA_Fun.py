from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import requests
import openai
import streamlit as st
from utils import generate

import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Example text

# read this pdf file
text = """We closed a solid first quarter, over delivering on our
commitment to profitable growth. In February, I made
several difficult, yet necessary decisions to drive better
focus, improve our ability to execute, and deliver an
accelerated path to meaningful profitability. We announced
a new operating model consisting of two business units, a
new leadership structure, a reduction in headcount and a
revised capital allocation plan that included a $1 billion
share repurchase program. With each week that passes, we are starting to yield more
results from these operational and organizational shifts and I am more confident than
ever that these decisions were the right ones for the long-term health of the business."""

# Function to extract named entities using OpenAI embeddings


def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        # Get the OpenAI embedding for the named entity
        embedding = generate.get_embeddings(ent.text, embedding_type='text')
        embedding = embedding[0]['embedding']
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "embedding": embedding
        })
    return entities


# Extract named entities with embeddings
named_entities = extract_entities(text)

# Display the extracted named entities
for entity in named_entities:
    st.write(f"Text: {entity['text']}")
    st.write(f"Label: {entity['label']}")
    st.write()


# Assume you have a matrix of embeddings called 'embeddings'
# Each row in the matrix represents an embedding vector
# get the embeddings from the named entities
embeddings = [entity['embedding'] for entity in named_entities]


# Convert embeddings to a numpy array
embeddings = np.array(embeddings)

# Apply dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Visualize the embeddings in a scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.6)
plt.title('Embeddings Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

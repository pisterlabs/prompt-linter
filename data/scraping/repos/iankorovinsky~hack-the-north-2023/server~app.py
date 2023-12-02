import soundfile as sf
import pandas as pd
import umap.umap_ as umap
from transformers import pipeline
import cv2
import csv
import seaborn as sns
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cohere
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='http://localhost:3000')
co = cohere.Client('COHERE_CLIENT')

transcription_map = {}
with open('df.csv', newline='') as csvfile:
    fragments = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(fragments)
    for row in fragments:
        id, filename, transcription, context = row
        transcription_map[transcription] = filename

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
transcriptions = list(transcription_map.keys())
embeddings = model.encode(transcriptions)

embeddings = {transcription_map[paragraph]: embedding for paragraph, embedding in zip(transcriptions, embeddings)}
embeddings.values()

reducer = umap.UMAP()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(embeddings.values()))
reduced_data = reducer.fit_transform(scaled_data)

context_map = {}
with open('df.csv', newline='') as csvfile:
    fragments = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(fragments)
    for row in fragments:
        id, filename, transcription, context = row
        context_map[context] = filename

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
contexts = list(context_map.keys())
context_embeddings = model.encode(contexts)

context_embeddings = {context_map[paragraph]: embedding for paragraph, embedding in zip(contexts, context_embeddings)}
context_embeddings.values()

c_scaled_data = scaler.fit_transform(list(context_embeddings.values()))
reduced_context_data = reducer.fit_transform(c_scaled_data)

@app.route('/api/branch', methods=["POST", "GET"])
def branch():
    query = request.form['query']
    user_input_embedding = model.encode(query)

    # Ensure user_input_embedding has the same number of dimensions as embeddings
    user_input_embedding = user_input_embedding.reshape(1, -1)

    transcription_similarities = cosine_similarity(user_input_embedding, list(embeddings.values()))
    context_similarities = cosine_similarity(user_input_embedding, list(context_embeddings.values()))

    transcription_weight = 0.7
    context_weight = 0.3

    max_len = max(len(transcription_similarities[0]), len(context_similarities[0]))
    transcription_similarities = np.pad(transcription_similarities, ((0, 0), (0, max_len - len(transcription_similarities[0]))), 'constant')
    context_similarities = np.pad(context_similarities, ((0, 0), (0, max_len - len(context_similarities[0]))), 'constant')

    combined_similarities = (transcription_weight * transcription_similarities + context_weight * context_similarities)

    most_similar_indices = combined_similarities.argsort()[0][::-1]
    most_similar_indices = most_similar_indices[:5]

    ARTICLE = []

    final_similarities = []
    df = pd.read_csv('df.csv')
    for idx in most_similar_indices:
        info = df.iloc[idx]
        ARTICLE.append(info.to_dict().get('Transcription'))
        ARTICLE.append(info.to_dict().get('Context'))
        final_similarities.append(info.to_dict())
    
    co_summary = co.summarize(text=' '.join(ARTICLE))

    return jsonify({'results': final_similarities, 'summary': co_summary })


if __name__ == '__main__':
    app.run(debug=True, port=2000)

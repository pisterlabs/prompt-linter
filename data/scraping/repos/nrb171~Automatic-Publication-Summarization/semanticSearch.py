import openai
import os
import tiktoken
import pandas as pd
from scipy import spatial
import numpy as np
import argparse
from flask import Flask, render_template, request, send_from_directory, session
import json
from flaskext.markdown import Markdown
import webbrowser



app = Flask(__name__)
Markdown(app)  # This initializes the Flask-Markdown extension.
app.secret_key = 'PeterJacksonIsTheBestDirectorOfAllTime'
def similaritySearch(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) :
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    embeddingModel = "text-embedding-ada-002"
    query_embedding_response = openai.Embedding.create(
        model=embeddingModel,
        input=query
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = []
    lastPaper = ""
    for i in range(len(df)):
        embedding = df["embedding"][i]
        embedding = embedding[1:-2].split(", ")
        embedding = [float(i) for i in embedding]
        relatedness = relatedness_fn(query_embedding, embedding)
        if df["name"][i] == lastPaper:
            if relatedness > strings_and_relatednesses[-1]["relatedness"]:
                strings_and_relatednesses[-1]["relatedness"] = max(strings_and_relatednesses[-1]["relatedness"], relatedness)
                strings_and_relatednesses[-1]["string"] = df["string"][i]
                continue
            else:
                continue
        
        strings_and_relatednesses.append({"name": df["name"][i][:-3], "relatedness": relatedness,"string": df["string"][i]})
        lastPaper = df["name"][i]

    # combine and average relatednesses for duplicate strings
    string_to_relatedness = {}
    for string_and_relatedness in strings_and_relatednesses:
        string = string_and_relatedness["string"]
        relatedness = string_and_relatedness["relatedness"]
        if string in string_to_relatedness:
            string_to_relatedness[string].append(relatedness)
        else:
            string_to_relatedness[string] = [relatedness]
        
    for string in string_to_relatedness:
        relatednesses = string_to_relatedness[string]
        string_to_relatedness[string] = np.mean(relatednesses)
    

    # sort by relatedness
    strings_and_relatednesses.sort(key=lambda x: x["relatedness"], reverse=True)
    return strings_and_relatednesses[:top_n]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    directory = request.form['directory']
    session['directory'] = directory
    top_n = int(request.form['top_n'])

    # Load data into array
    df = pd.read_csv(directory + "/.embeddings/embeddings.csv")

    # Get results
    similaritySearchResults = similaritySearch(query, df, top_n=top_n)

    return render_template('index.html', results=similaritySearchResults,prev_query=query, prev_directory=directory, prev_top_n=top_n)

@app.route('/<filename>', methods=['GET'])
def serve_pdf(filename):
    directory = session.get('directory')  # Assuming you store the directory in a session variable
    files = os.listdir(directory)
    file = files[files[0].split(".")[0][:-3] == filename.split(".")[0]]
    if not directory:
        return "Directory not set!", 400
    return send_from_directory(directory, file, as_attachment=False)

if __name__ == '__main__':
    embeddingModel = "text-embedding-ada-002"
    encodingName="cl100k_base"
    with open("key.ini", "r") as f:
        openai.api_key = f.read().strip()
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)




from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import openai
from openai import Embedding
from openai.embeddings_utils import distances_from_embeddings
import numpy as np
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("API key not found in environment variables")

# Load the existing embeddings CSV file
df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


def create_context(question, df, max_len=1800, size="ada"):
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    returns = []
    cur_len = 0
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        cur_len += row['n_tokens'] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])
    return "\n\n###\n\n".join(returns)

def answer_question(df, question, context, model="gpt-3.5-turbo-instruct", max_len=1800, size="ada", debug=False, max_tokens=150, stop_sequence=None):
    try:
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

# on accessing root url render the "index.html"
@app.route("/")
def index():
    return render_template("index.html")

# computes question embeddings, determines relevant context to generate answer
@app.route("/ask", methods=["POST"])
def ask():
    question = request.get_json().get("question")
    print("Received question:", question)
    question_embeddings = np.array(openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df['distances'] = distances_from_embeddings(question_embeddings, df['embeddings'].values, distance_metric='cosine')
    context = create_context(question, df)
    answer = answer_question(df, question=question, context=context)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
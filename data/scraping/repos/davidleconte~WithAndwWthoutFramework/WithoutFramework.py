#
# This script does the following:
#
# Extracts text from PDFs.
# Embeds the documents using Sentence Transformers.
# Sets up a RAG model from Hugging Face.
# Creates a SQLite database for context management.
# Provides a function for querying OpenAI.
# Creates a simple Flask application to interact with the user.
# You would need to replace 'your-api-key' and '/path/to/pdf/dir' with your actual OpenAI API key and the path to your PDF directory, respectively.

import os
import pdfplumber
import sqlite3
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Document Preparation
def extract_text_from_pdfs(pdf_dir):
    text_data = {}
    for subdir, _, files in os.walk(pdf_dir):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(subdir, file)
                with pdfplumber.open(file_path) as pdf:
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text()
                    text_data[file_path] = text
    return text_data

# Document Embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embed_documents(documents):
    embeddings = {path: model.encode(text) for path, text in documents.items()}
    return embeddings

# Setting up FAISS Index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(len(next(iter(embeddings.values()))))  # Assuming all embeddings have the same length
    faiss.normalize_L2(embeddings)
    index.add(np.array(list(embeddings.values())))
    return index

# Setting up RAG
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

def ask_rag(question):
    inputs = tokenizer(question, return_tensors="pt")
    generated = rag_model.generate(**inputs)
    answer = tokenizer.decode(generated[0], skip_special_tokens=True)
    return answer

# Context Management
conn = sqlite3.connect('conversation_history.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS conversation
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_input TEXT,
              system_response TEXT,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

def store_conversation(user_input, system_response):
    c.execute("INSERT INTO conversation (user_input, system_response) VALUES (?, ?)", (user_input, system_response))
    conn.commit()

# Integration with OpenAI
openai.api_key = 'your-api-key'

def ask_openai(question):
    response = openai.Completion.create(
        engine="davinci",
        prompt=question,
        max_tokens=150
    )
    answer = response['choices'][0]['text'].strip()
    return answer

# Flask Application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        answer = ask_rag(question)  # Or ask_openai(question) depending on which you prefer
        store_conversation(question, answer)
        return render_template('index.html', answer=answer)
    return render_template('index.html')

if __name__ == '__main__':
    pdf_dir = '/path/to/pdf/dir'
    documents = extract_text_from_pdfs(pdf_dir)
    embeddings = embed_documents(documents)
    faiss_index = create_faiss_index(embeddings)
    app.run(debug=True)


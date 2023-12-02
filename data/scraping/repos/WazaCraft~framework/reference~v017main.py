import os
import requests
import openai
import datetime
import numpy as np
import pickle
import socket
import threading
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

openai_api_key = input("Please enter your OpenAI API key: ")
openai.api_key = openai_api_key
os.environ['USER_PROMPT'] = 'Here is the info from the text: {content}. Based on this, what is the answer to "{question}"?'

def chunk_text(text, max_tokens=8000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def get_embedding_for_large_text(text):
    chunks = chunk_text(text)
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
        embedding = response['data'][0]['embedding']
        embeddings.append(embedding)
    return embeddings

def create_file_name(url, extension='txt'):
    parsed_url = urlparse(url)
    url_path_parts = parsed_url.path.strip('/').split('/')
    last_part = url_path_parts[-1] if url_path_parts else parsed_url.netloc
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"{last_part}-{current_date}.{extension}"

def get_most_similar_text_chunk(question, embeddings_dict):
    question_embedding = get_embedding_for_large_text(question)[0]
    similarity_scores = []
    for text_chunk_embedding in embeddings_dict['embeddings']:
        similarity_scores.append(cosine_similarity([question_embedding], [text_chunk_embedding])[0][0])
    most_similar_index = np.argmax(similarity_scores)
    return embeddings_dict['text_chunks'][most_similar_index]

def generate_response(question, embeddings_dict):
    similar_text_chunk = get_most_similar_text_chunk(question, embeddings_dict)
    user_prompt = os.environ['USER_PROMPT'].format(content=similar_text_chunk, question=question)
    messages = [
        {"role": "system", "content": "You are a knowledgeable assistant."},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        assistant_reply = response['choices'][0]['message']['content']
        return assistant_reply
    except Exception as e:
        return str(e)

def extract_and_save_urls(html_content, file):
    soup = BeautifulSoup(html_content, 'html.parser')
    for link in soup.find_all('a'):
        url = link.get('href')
        if url:
            file.write(url + '\n')

def save_embeddings_to_file(embeddings_dict, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(embeddings_dict, file)

def load_embeddings_from_file(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

embeddings_dict = {}

url = 'https://www.rssground.com/services/rss-converter/64a0a74cd5ee7/RSS-Payload'
response = requests.get(url)
text = response.text
file_name = create_file_name(url)

with open(file_name, 'w') as file:
    file.write(text)
    extract_and_save_urls(text, file)

embeddings = get_embedding_for_large_text(text)
chunks = chunk_text(text)
embeddings_file_name = create_file_name(url, extension='pkl')
embeddings_dict[embeddings_file_name] = {'text_chunks': chunks, 'embeddings': embeddings}
save_embeddings_to_file(embeddings_dict, embeddings_file_name)

print("Daily data refreshed. Now browsing 75+ deal feeds.")

@app.route('/ask', methods=['GET'])
def ask_question():
    question = request.args.get('question')
    if question:
        responses = []
        for embeddings_file_name in embeddings_dict.keys():
            response = generate_response(question, embeddings_dict[embeddings_file_name])
            responses.append(response)
        return jsonify(responses)
    return jsonify({"error": "No question provided"})

def run_web_api(port):
    app.run(port=port)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

api_thread = None

while True:
    user_input = input("Enter URL or question or 'deal-id up' or 'deal-id down' (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'deal-id up':
        if api_thread is None or not api_thread.is_alive():
            port = 5000
            while is_port_in_use(port):
                port = int(input(f"Port {port} is in use. Please enter a different port: "))
            api_thread = threading.Thread(target=run_web_api, args=(port,))
            api_thread.daemon = True
            api_thread.start()
        else:
            print("Server is already running")

    elif user_input.lower() == 'deal-id down':
        if api_thread and api_thread.is_alive():
            print("Stopping the server.")
            requests.post(f'http://localhost:{port}/shutdown')
            api_thread.join()
        else:
            print("Server is not running")

    elif user_input.lower().startswith('http'):
        url = user_input
        response = requests.get(url)
        text = response.text
        file_name = create_file_name(url)

        with open(file_name, 'w') as file:
            file.write(text)
            extract_and_save_urls(text, file)

        embeddings = get_embedding_for_large_text(text)
        chunks = chunk_text(text)
        embeddings_file_name = create_file_name(url, extension='pkl')
        embeddings_dict[embeddings_file_name] = {'text_chunks': chunks, 'embeddings': embeddings}
        save_embeddings_to_file(embeddings_dict, embeddings_file_name)

    else:
        question = user_input
        for embeddings_file_name in embeddings_dict.keys():
            response = generate_response(question, embeddings_dict[embeddings_file_name])
            print(response)

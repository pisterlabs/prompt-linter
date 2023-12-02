import urllib.request
import fitz
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, util
import torch
import shutil
from langchain.document_loaders import PyPDFLoader
import tensorflow_hub as hub

"""
più piccolo è il chunk meno token vengono utilizzati nel fare la domanda || configurazione per singolo file: chunk_size = 150 n_chunks = 5 tokens = 750
"""
chunk_size = 50
n_chunks = 15
#model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
model = SentenceTransformer("bert model")
from langchain.embeddings import CohereEmbeddings
#model = CohereEmbeddings(model="embed-english-light-v2.0", cohere_api_key="my-api-key")
"""Metodi per la manipolazione del testo; conversione da pdf a text"""

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(path):
    text = []
    if isinstance(path, list):
        total_pages = 0
        for file in path:
            loader = PyPDFLoader(file.name)
            pages = loader.load_and_split()
            for page in pages:
                page = page.page_content
                page = preprocess(page)
                text.append(page)
    else:
        loader = PyPDFLoader(path.name)
        pages = loader.load_and_split()
        for page in pages:
            page = page.page_content
            page = preprocess(page)
            text.append(page)
    return text

def text_to_chunks(texts, word_length=chunk_size, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i + word_length]
            if (i + word_length) > len(words) and (len(chunk) < word_length) and (
                    len(text_toks) != (idx + 1)):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx + start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks

"""La classe SemanticSearch è l'attore che andrà a generare l'embedding e calcolare i chunk di testo più vicini alla domanda dell'utente"""

class SemanticSearch:

    def __init__(self):
        self.fitted = False

    def encode(self, chunk):
        embeddings = model.encode(chunk, convert_to_tensor=True)
        return embeddings

    # applico il nearest neighbors sull'embedding del pdf
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data  # salvo i chunks del pdf in data
        self.corpus_embeddings = self.get_text_embedding(data, batch=batch)  # qui creo gli embedding
        self.fitted = True

    def fit2(self, data, embeddings_file, n_neighbors=5):
        self.data = data  # salvo i chunks del pdf in data
        self.dim_corpus = len(data)
        self.corpus_embeddings = np.load(embeddings_file)  # qui creo gli embedding
        self.fitted = True

    # restituisco i top n chunks più simili alla domanda
    def __call__(self, text): # text è la domanda input dell'utente
        top_k = min(n_chunks, self.dim_corpus)
        domanda_embeddings = self.encode([text]) # embedding applicato alla domanda
        cos_scores = util.cos_sim(domanda_embeddings, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return [self.data[i] for i in top_results[1]]

    # questa è la classe che fa l'embedding sul contenuto del pdf
    def get_text_embedding(self, texts, batch):
        embeddings = []
        print("Numero di chunks nel documento: ")
        print(len(texts))
        self.dim_corpus = len(texts)

        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i + batch)]

            emb_batch = self.encode(text_batch)  # chiamo il modello
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)  #a function provided by NumPy that takes a sequence of arrays and stacks them vertically to form a new array.
        return embeddings

"""Funzioni per interagire con le API di quei tirchi della openAI"""

# dato un object semantic search e, se non è già stato fatto, creo gli embeddings del pdf ed applico il nearest neighbors
def load_recommender(path, flagUrl):
    start_page = 1
    global recommender
    if len(path) > 1:
        embeddings_file = f"docs\multiFileEmbeddings.npy"
    elif(flagUrl):
        pdf_file = os.path.basename(path)
    else:
        path = path[0]
        pdf_file = os.path.basename(path.name)
        embeddings_file = f"docs\{pdf_file}_{start_page}.npy"

    texts = pdf_to_text(path)
    chunks = text_to_chunks(texts, start_page=start_page)
    # fitto il reccomender
    # se ho già il file fitto direttamente caricandolo
    if os.path.isfile(embeddings_file):
        recommender.fit2(chunks, embeddings_file)
        return "Embeddings loaded from file"
    recommender.fit(chunks)
    np.save(embeddings_file, recommender.corpus_embeddings)
    print("sto salvando roba")
    return 'Corpus Loaded.'

def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=500,     #incide sulla lunghezza della risposta output, quindi sul prezzo della chiamata
        n=1,
        stop=None,
        temperature=0.7,    #più è basso meno si spende, forse
    )
    message = completions.choices[0].text
    return message

def generate_text2(openAI_key, prompt, engine="gpt-3.5-turbo-0301"):
    openai.api_key = openAI_key

    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7
    )

    message = response.choices[0].message['content']
    return message

# per il debug vai qui
def generate_answer(question, openAI_key):
    # genero i chunks
    topn_chunks = recommender(question) # metodo __call__ : confronto l'embedding della domanda all'embedding del pdf ed ottengo gli n snippet di testo più vicini
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. " \
              "If the search results mention multiple subjects with the same name, create separate answers for each. " \
              "Only include information found in the results and don't add any additional information. " \
              "Make sure the answer is correct and don't output false content. " \
              "If the text does not relate to the query, simply state 'Non è stata trovata una risposta alla tua domanda nel testo'." \
              "Ignore outlier search results which has nothing to do with the question. Only answer what is asked. " \
              "The answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"

    """
    file_object = open('docs\domande.txt', 'a')
    file_object.write('\n\n\nprompt:\n')
    file_object.write(prompt)
    # Close the file
    file_object.close()
    """

    #answer = generate_text(openAI_key, prompt, "text-davinci-003")
    answer = prompt
    return answer

#1.
# prima faccio l'embedding sul pdf, poi anche alla domanda e infine ricerca semantica
def question_answer(url, file, question, openAI_key):
    if url.strip() != '':
        glob_url = url
        download_pdf(glob_url, 'docs\corpus.pdf')
        load_recommender('docs\corpus.pdf', True)  # creo gli embeddings del pdf ed applico il nearest neighbors
    else:
        load_recommender(file, False)

    if question.strip() == '':
        return '[ERROR]: Question field is empty'

    return generate_answer(question, openAI_key)

def cleanup():
    embeddings_file = f"docs\multiFileEmbeddings.npy"
    if os.path.isfile(embeddings_file):
        os.remove(embeddings_file)

    # WINDOWS
    directory = "C:\\Users\\xlits\\AppData\\Local\\Temp"  # Replace with the target directory

    # List all directories in the specified directory
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name.startswith("tmp"):
                folder_path = os.path.join(root, dir_name)
                try:
                    # Delete the folder and its contents
                    shutil.rmtree(folder_path)
                    print(f"Folder '{folder_path}' deleted successfully.")
                except OSError as e:
                    print(f"Failed to delete folder '{folder_path}': {e}")



recommender = SemanticSearch()
cleanup()
title = 'Chat Insights'
description = """Team Namec spacca."""

with gr.Blocks(theme='freddyaboulton/dracula_revamped') as demo:
    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():
        with gr.Group():
            gr.Markdown(
                f'<p style="text-align:center">Get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a></p>')
            openAI_key = gr.Textbox(label='Enter your OpenAI API key here')
            url = gr.Textbox(label='Enter PDF URL here')
            gr.Markdown("<center><h4>OR<h4></center>")
            file = gr.File(label='Upload your PDF', file_types=['.pdf'], file_count="multiple")
            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')

        btn.click(question_answer, inputs=[url, file, question, openAI_key], outputs=[answer])
demo.launch(debug=True)
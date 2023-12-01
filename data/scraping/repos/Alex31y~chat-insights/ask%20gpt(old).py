import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors


def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
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


class SemanticSearch:

    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False

    # applico il nearest neighbors sull'embedding del pdf
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data  # salvo i chunks del pdf in data
        self.embeddings = self.get_text_embedding(data, batch=batch)  # qui creo gli embedding
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def fit2(self, data, embeddings_file, n_neighbors=5):
        self.data = data  # salvo i chunks del pdf in data
        self.embeddings = np.load(embeddings_file)  # qui creo gli embedding
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    # quando la classe viene usata come metodo, usufruendo del nn, confronto l'embedding della domanda all'embedding del pdf
    def __call__(self, text): # text è la domanda input dell'utente
        embeddings = self.use([text]) # universal-sentence-encoder applicato alla domanda
        neighbors = self.nn.kneighbors(embeddings, return_distance=False)[0] # chiamo il nearest_neighbors già fittato sulla domanda, restituisce i vicini i.e. [33 5 21 14 25]
        return [self.data[i] for i in neighbors]

    # questa è la classe che fa l'embedding sul contenuto del pdf
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        print("text len : ")
        print(len(texts))

        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i + batch)]

            emb_batch = self.use(text_batch)  # chiamo l'hub di google sul text
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


# def load_recommender(path, start_page=1):
#    global recommender
#   texts = pdf_to_text(path, start_page=start_page)
#   chunks = text_to_chunks(texts, start_page=start_page)
#    recommender.fit(chunks)
#    return 'Corpus Loaded.'


# dato un object semantic search e, se non è già stato fatto, creo gli embeddings del pdf ed applico il nearest neighbors
def load_recommender(path, start_page=1):
    global recommender
    pdf_file = os.path.basename(path)
    embeddings_file = f"{pdf_file}_{start_page}.npy"

    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    # fitto il reccomender
    # se ho già il file fitto direttamente caricandolo
    if os.path.isfile(embeddings_file):
        recommender.fit2(chunks, embeddings_file)
        return "Embeddings loaded from file"
    recommender.fit(chunks)
    np.save(embeddings_file, recommender.embeddings)
    return 'Corpus Loaded.'


def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_text2(openAI_key, prompt, engine="gpt-3.5-turbo-0301"):
    openai.api_key = openAI_key
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]

    completions = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].message['content']
    return message

# 2
def generate_answer(question, openAI_key):
    # genero i chunks
    topn_chunks = recommender(question) # metodo __call__ : confronto l'embedding della domanda all'embedding del pdf ed ottengo gli n snippet di testo più vicini
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. " \
              "Cite each reference using [ Page Number] notation (every result has this number at the beginning). " \
              "Citation should be done at the end of each sentence. If the search results mention multiple subjects " \
              "with the same name, create separate answers for each. Only include information found in the results and " \
              "don't add any additional information. Make sure the answer is correct and don't output false content. " \
              "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier " \
              "search results which has nothing to do with the question. Only answer what is asked. The " \
              "answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(openAI_key, prompt, "text-davinci-003")

    file_object = open('domande.txt', 'a')
    file_object.write('\n\n\nprompt:\n')
    file_object.write(prompt)
    # Close the file
    file_object.close()
    return answer

#1.
# prima faccio l'embedding sul pdf, poi anche alla domanda e applico il knn
def question_answer(url, file, question, openAI_key):
    if url.strip() != '':
        glob_url = url
        download_pdf(glob_url, 'corpus.pdf')
        load_recommender('corpus.pdf')  # creo gli embeddings del pdf ed applico il nearest neighbors

    else:
        load_recommender(file)

    if question.strip() == '':
        return '[ERROR]: Question field is empty'

    return generate_answer(question, openAI_key)


recommender = SemanticSearch()

key = ""
domanda = "quando scade la candidatura per partecipare al bando?"
file_object = open('domande.txt', 'a')
file_object.write('\n')
file_object.write("domanda: \n")
file_object.write(domanda)

answ = question_answer("", "erasmus.pdf", domanda, key)

file_object.write("risposta: \n")
file_object.write(answ)
# Close the file
file_object.close()
print(domanda)
print(answ)

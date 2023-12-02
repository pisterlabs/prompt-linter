""" create embeddings """
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from numpy.linalg import norm
import pickle
import openai
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)


# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text):
    text = text.replace('\n', ' ')
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    # np.array
    return response['data'][0]['embedding']  # Returns the embedding vector


def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


def count_tokens(text:str, model:str = "gpt-4") -> int:
    """ funkcja zlicza tokeny """
    num_of_tokens = 0
    enc = tiktoken.encoding_for_model(model)
    num_of_tokens = len(enc.encode(text))

    return num_of_tokens


# -------------------------------- MAIN ----------------------------------------
if __name__ == '__main__':

    # pomiar czasu wykonania
    start_time = time.time()

    # embeddigns dla pytania
    query_text = "miejsce urodzenia, miejsce śmierci (gdzie zmarł), miejsce pochówku / pogrzebu, data urodzenia (kiedy się urodził), data śmierci (kiedy zmarł) - na przykład z podanego w biogramie zakresu lat (1790-1800), data pochówku, pogrzebu"
    query_path = Path("..") / 'emb_psb_250' / 'basic_query.pkl'

    query_text = "wyszukaj wszystkich krewnych lub powinowatych głównego bohatera/bohaterki tekstu. Możliwe rodzaje pokrewieństwa: ojciec, matka, syn, córka, brat, siostra, żona, mąż, teść, teściowa, dziadek, babcia, wnuk, wnuczka, szwagier, szwagierka, siostrzeniec, siostrzenica, bratanek, bratanica, kuzyn, kuzynka, zięć, synowa."
    query_path = Path("..") / 'emb_psb_250' / 'relations_query.pkl'
    
    if os.path.exists(query_path):
        with open(query_path, "rb") as pkl_file:
            query_embedding = pickle.load(file=pkl_file, encoding='utf-8')
    else:
        query_embedding = get_embedding(query_text)
        with open(query_path, "wb") as pkl_file:
            pickle.dump(query_embedding, pkl_file)

    # dane z pliku tekstowego
    #data_file = Path("..") / "emb_psb_250" / "Curie_Maria.pkl"
    data_file = Path("..") / "emb_psb_250" / 'Stanislaw_August_Poniatowski.pkl'
    with open(data_file, "rb") as pkl_file:
        biogram_embedding = pickle.load(file=pkl_file, encoding='utf-8')

    sent_similarity = []
    for key, value in biogram_embedding.items():
        sent, sent_vector = value
        similarity = cosine_similarity(query_embedding, sent_vector)
        sent_similarity.append((sent, similarity, key))
        print(f'{similarity}@{sent}')

    # sortowanie wg podobieństwa
    sent_similarity.sort(key=lambda x: x[1], reverse=True)

    best_sent = []
    max_tokens = 3000

    # ograniczenie tekstu do maksymalnej liczby tokenów na podstawie podobieństwa
    # zdań do pytania
    tokens_sum = 0
    for item in sent_similarity:
        sent_tokens = count_tokens(item[0])
        if tokens_sum + sent_tokens < max_tokens:
            best_sent.append(item)
            tokens_sum += sent_tokens
        else:
            break

    output_text = ''
    best_sent.sort(key=lambda x: x[2])
    for b_item in best_sent:
        output_text += b_item[0].replace('\n',' ') + ' '

    # skrócony biogram
    #print(output_text)

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')

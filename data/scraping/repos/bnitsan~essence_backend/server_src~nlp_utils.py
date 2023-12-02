from sklearn.metrics.pairwise import cosine_similarity
import nltk
from langdetect import detect
from nltk.tokenize import sent_tokenize
import requests
import numpy as np
import os
import yaml
from cachelib.file import FileSystemCache
import hashlib
import re
from . import general_utils
from retry import retry

import openai as openaiembed


with open("server_src/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg = cfg["config"]

SENT_TOKEN_PROTECTED = cfg["SENT_TOKEN_PROTECTED"]
MIN_SENTENCE_LEN_QA_EMBED = cfg["MIN_SENTENCE_LEN_QA_EMBED"]
MAX_SENTENCE_LEN_QA_EMBED = cfg["MAX_SENTENCE_LEN_QA_EMBED"]
SENTENCE_QA_EMBED_MODEL = cfg["SENTENCE_QA_EMBED_MODEL"]
CACHE_QA_SECONDS = cfg["CACHE_QA_SECONDS"]
CACHE_QA_THRESHOLD = cfg["CACHE_QA_THRESHOLD"]
INF_ENDPOINT_SENT_TRANS = cfg["INF_ENDPOINT_SENT_TRANS"]
COMPLETION_TIMEOUT = cfg["COMPLETION_TIMEOUT"]

SECRET_SENTEMBED_KEY = os.getenv("SECRET_HF_MODEL_KEY") if os.getenv("SECRET_HF_MODEL_KEY") else ''

data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data'))  # get absolute path to one folder up
if os.getenv("ESSENCE_DATA_PATH"):
    data_path = os.getenv("ESSENCE_DATA_PATH")
embed_cache = FileSystemCache(os.path.join(data_path, 'embed_cache'), threshold=CACHE_QA_THRESHOLD, default_timeout=CACHE_QA_SECONDS)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openaiembed.api_key = OPENAI_API_KEY
openaiembed.api_type = 'openai'

azure_flag = False
if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
    openaiembed.api_type = "azure"
    openaiembed.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
    openaiembed.api_version = "2023-05-15"
    openaiembed.api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_flag = True

def hash_text(text):
    sha256 = hashlib.sha256()
    sha256.update(text.encode())
    return sha256.hexdigest()

def find_islands(indices, sigma: int, length: int):
    '''
    This function takes indices that list to locations of the top sentences 
    fitting the query. It also takes sigma, a maximum distance around an index that should be included in the island.
    When islands touch or overlap, they are merged into one island.
    length: maximal index value + 1
    '''
    if not isinstance(sigma, int) or sigma < 0:
        raise ValueError('sigma must be a non-negative integer')
    if len(indices) < 1:
        return []

    indices.sort()
    islands = [[i for i in range(indices[0]-sigma, indices[0]+sigma+1) if (i >= 0 and i < length)]]

    for j in range(1, len(indices)):
        if indices[j] - indices[j-1] <= (2*sigma+1):
            islands[-1].extend([i for i in range(islands[-1][-1]+1, indices[j]+sigma+1) if (i >= 0 and i < length)])
        else:
            islands.append([i for i in range(indices[j]-sigma, indices[j]+sigma+1) if (i >= 0 and i < length)])

    return islands

def post_request(url, data):
    response_post = requests.post(url, json=data)
    return response_post.json()

def multiple_replace(text: str, replacements: dict) -> str:
    ''' Thanks ChatGPT for this function! 
    Replace multiple substrings of a string with another substring.
    replacements is a dictionary of {str_to_find: str_to_replace_with}
    '''
    # Compile a regular expression pattern that matches all the substrings
    # to be replaced and capture them as groups
    pattern = re.compile("|".join("(%s)" % re.escape(key) for key in replacements.keys()))

    # Use the sub function to replace all the occurrences of the captured groups
    # in the text with their corresponding replacements
    return pattern.sub(lambda x: replacements[x.group(0)], text)

def prepare_text_for_sent_split(text):
    pairs = {'Fig.': 'Figure', 'FIG.': 'Figure', 'Figs.': 'Figures', 'FIGS.': 'Figures',
             'Sec.': 'Section', 'SEC.': 'Section', 'Secs.': 'Sections', 'SECS.': 'Sections',
             'Eq.': 'Equation', 'EQ.': 'Equation', 'Eqs.': 'Equations', 'EQS.': 'Equations',
             'Ref.': 'Reference', 'REF.': 'Reference', 'Refs.': 'References', 'REFS.': 'References',
             'in App.': 'in Appendix', 'In App.': 'In Appendix', 'in APP.': 'in Appendix', 'In APP.': 'In Appendix'}
    text = multiple_replace(text, pairs)
    text = multiple_replace(text, SENT_TOKEN_PROTECTED)
    return text

def rerun_text_after_sent_split(sentences):
    SENT_TOKEN_PROTECTED_INV = {v: k for k, v in SENT_TOKEN_PROTECTED.items()}
    sentences = [multiple_replace(sentence, SENT_TOKEN_PROTECTED_INV) for sentence in sentences]
    return sentences

def quality_assurance_sentences(sentences, min_sentence_length=MIN_SENTENCE_LEN_QA_EMBED, max_sentence_length=MAX_SENTENCE_LEN_QA_EMBED):
    return [sentence for sentence in sentences if len(sentence) >= min_sentence_length and len(sentence) <= 2000]

def set_embed(id: str, text: str, backend: str, embeddings: list):
    embed_cache.add(id, {"text": text, "backend": backend, "embeddings": embeddings})

def get_embed_if_exists(id: str, text: str, backend: str):
    if embed_cache.has(id):
        elem = embed_cache.get(id)
        if elem["text"] == text and elem["backend"] == backend:
            print('Going to use cached embeddings...')
            return elem["embeddings"]
    return None

@retry(exceptions=openaiembed.error.Timeout, tries=4)
def OpenAIEmbeddings(input, model=SENTENCE_QA_EMBED_MODEL):
    if azure_flag:
        print('Using Azure OpenAI...')
        if len(input) == 1:
            openai_embeddings = openaiembed.Embedding.create(input=input, engine="essence-embed")
        else:
            # Azure OpenAI, as of May 22, 2023, does not support batch embeddings. Sad.
            openai_embeddings = {}
            openai_embeddings["data"] = []
            for i in range(len(input)):
                openai_embeddings["data"].append(openaiembed.Embedding.create(input=[input[i]], engine="essence-embed")["data"][0])
            return openai_embeddings
    else:
        print('Using OpenAI... with model: ' + model)
        openai_embeddings = openaiembed.Embedding.create(input=input, model=model, request_timeout=COMPLETION_TIMEOUT)
        print('Finished.')
    return openai_embeddings

def get_single_embedding(string, backend="openai"):
    if backend == "sent_trans":
        response = post_request(INF_ENDPOINT_SENT_TRANS + '/predict', {'sentences': [string], 'secret_key': SECRET_SENTEMBED_KEY})
        embedding = response["embeddings"]
    elif backend == "openai":
        openai_embeddings = OpenAIEmbeddings([string], model=SENTENCE_QA_EMBED_MODEL) # openai.Embedding.create(input = [string], model=SENTENCE_QA_EMBED_MODEL)
        embeddings = openai_embeddings["data"][0]["embedding"]
        embedding = [embeddings]
    else:
        raise ValueError('backend not supported')
    return embedding

def get_embeddings_similarity(emb1, emb2):
    cosine_sim = cosine_similarity(emb1, emb2).flatten().tolist()
    return cosine_sim

def combine_strings(l, m):
    print('Shortening the text by combining sentences... m =', m)
    if m < 2: return l
    result = []
    for i in range(0, len(l), m):
        combined = ' '.join(l[i:i + m])
        result.append(combined)
    return result

# @general_utils.retry_on_timeout(retries=3, timeout_seconds=15)
def get_embeddings(question: str, text: str, url:str, backend="openai", max_sentences=100, compact_sentences=1):
    text = prepare_text_for_sent_split(text)
    sentences = sent_tokenize(text)
    sentences = rerun_text_after_sent_split(sentences)
    sentences = quality_assurance_sentences(sentences)

    if compact_sentences > 1:
        sentences = combine_strings(sentences, compact_sentences)
        
    # we'd like to reduce the number of sentences to max_sentences. We do it by batching to nearest power of 2.
    # sentences = combine_strings(sentences, 2 ** int(np.floor(np.log2(len(sentences) / max_sentences))))
    # currently inactive.

    if len(sentences) == 0:
        print('ERROR: NO SENTENCES FOUND IN THE TEXT.')
        raise ValueError('No sentences found in text.')

    if backend == "sent_trans":
        response_q = post_request(INF_ENDPOINT_SENT_TRANS + '/predict', {'sentences': [question], 'secret_key': SECRET_SENTEMBED_KEY})
        embeddings_q = response_q["embeddings"]

        cache_embed_response = get_embed_if_exists(url + hash_text(text), text, backend)
        if cache_embed_response is not None:
            embeddings_a = cache_embed_response
        else:
            response_a = post_request(INF_ENDPOINT_SENT_TRANS + '/predict', {'sentences': sentences, 'secret_key': SECRET_SENTEMBED_KEY})
            embeddings_a = response_a["embeddings"]
            set_embed(url + hash_text(text), text, backend, embeddings_a)

    elif backend == "openai":
        print('Going to use OpenAI embeddings...')
        openai_embeddings_q = OpenAIEmbeddings([question], model=SENTENCE_QA_EMBED_MODEL)
        if "data" not in openai_embeddings_q:
            print('ERROR: OPENAI EMBEDDINGS API FAILED.')
            raise ValueError('OpenAI Embeddings API failed.')
        embeddings_q = openai_embeddings_q["data"][0]["embedding"]
        embeddings_q = [embeddings_q]

        cache_embed_response = get_embed_if_exists(url + hash_text(text), text, backend)
        if cache_embed_response is not None:
            embeddings_a = cache_embed_response
        else:
            openai_embeddings_a = OpenAIEmbeddings(sentences, model=SENTENCE_QA_EMBED_MODEL)
            if "data" not in openai_embeddings_q:
                raise ValueError('OpenAI Embeddings API failed.')
            embeddings_a = [openai_embeddings_a["data"][i]["embedding"] for i in range(len(sentences))]
            set_embed(url + hash_text(text), text, backend, embeddings_a)
    else:
        raise ValueError('backend not supported')
    cosine_sim = get_embeddings_similarity(embeddings_q, embeddings_a)
    
    return cosine_sim, sentences, embeddings_a, embeddings_q

def get_most_matching_sentences_to_answer(answer: str, embeddings, top=4):
    answer_embeddings = get_single_embedding(answer)
    similarities = get_embeddings_similarity(answer_embeddings, embeddings)
    
    return similarities

def get_supporting_sentences(sentences_islands, embeddings_a, answer, sentences, top_answers):
    candidate_sentences_locs = [i for island in sentences_islands for i in island]
    candidate_embeddings = [embeddings_a[i] for i in candidate_sentences_locs]
    cosine_sim_answer = get_most_matching_sentences_to_answer(answer, candidate_embeddings)
    
    top_locs = np.sort(np.argsort(cosine_sim_answer)[-top_answers:])

    top_locs_islands = [[top_locs[0]]]
    for i in range(1, len(top_locs)):
        if top_locs[i] - top_locs[i-1] == 1:
            top_locs_islands[-1].append(top_locs[i])
        else:
            top_locs_islands.append([top_locs[i]])
    supporting_sentences = [' '.join([sentences[candidate_sentences_locs[i]] for i in sublist]) for sublist in top_locs_islands]
    return supporting_sentences

def text_not_in_english(text):
    # use NLTK to check if text is in English by a simple heuristic
    try:
        if (detect(text[:150]) == 'en' or detect(text[len(text)//2:len(text)//2+150]) == 'en' or detect(text[-150:]) == 'en'):
            return False
        if (detect(text[:500]) == 'en' or detect(text[len(text)//2:len(text)//2+500]) == 'en' or detect(text[-500:]) == 'en'):
            return False
    except Exception as e:
        # if language detection fails -- possibly throws LangDetectException -- assume it is not in English
        print(e)
        return True
    return True

def detect_language(text):
    return 'en' if not text_not_in_english(text) else 'non-en'

def clean_marked_text(marked_text, min_length=3):
    # remove all lines with less than min_length characters
    marked_text = '\n'.join([line for line in marked_text.split('\n') if len(line) >= min_length])
    return marked_text

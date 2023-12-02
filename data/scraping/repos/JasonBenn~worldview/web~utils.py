import errno
import os
import re
import signal
import subprocess
from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import List
from typing import Tuple

import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import numpy as np
import spacy
from django.utils import timezone
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from nltk.corpus import stopwords
from tqdm import tqdm

from worldview import settings


def group_by(iterable, key):
    grouped = defaultdict(list)
    for x in iterable:
        grouped[key(x)].append(x)
    return {k: list(v) for k, v in grouped.items()}


def now() -> datetime:
    return timezone.now()


def clean_title(string: str) -> str:
    return "".join([c for c in string.replace(' ', '-') if c.isalnum() or c == '-']).lower()


def remove_newlines(string: str) -> str:
    return string.replace('\n', '<br>')


def asciify(text: str) -> str:
    return re.sub(r'[^\x00-\x7f]', r'', text)


def get_text_chunks(text: str) -> List[str]:
    chunks = []
    # TODO: cut on token boundary: https://spacy.io/usage/linguistic-features#retokenization
    for i in range(len(text) // 512 + 1):
        start_index = i * 512
        end_index = (i + 1) * 512
        chunks.append(text[start_index:end_index])
    return chunks


class TimeoutError(Exception):
    pass


def timeout(seconds=5, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def cosine_distance(a: np.array, b: np.array) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def flatten(arr):
    for i in arr:
        if isinstance(i, list):
            yield from flatten(i)
        else:
            yield i


def plot_and_show(filename: str = 'test'):
    plt.show()
    dirpath = f"{settings.BASE_DOCUMENT_DIR}/plots"
    os.makedirs(dirpath, exist_ok=True)
    filepath = f"{dirpath}/{filename}.png"
    plt.savefig(filepath)
    # For some reason, I have to interrupt this, and afterwards the png is empty.
    # subprocess.call(["open", filepath])


def get_stop_words():
    stop_words = stopwords.words('english')
    stop_words.extend(
        ['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go',
         'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily',
         'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may',
         'take', 'come'])
    return stop_words


def make_topic_model(words, num_topics: int = 20) -> Tuple[LdaModel, float]:
    # Create Dictionary
    id2word = corpora.Dictionary(words)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in words]
    # Build LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=10,
        passes=10,
        alpha='symmetric',
        iterations=100,
        per_word_topics=True
    )
    coherence_score = CoherenceModel(model=lda_model, texts=words, dictionary=id2word, coherence='c_v').get_coherence()
    return lda_model, coherence_score


def preprocess_docs_to_words(docs: List[str]) -> List[List[str]]:
    docs_as_words = [sent_to_words(x) for x in docs]
    lemmatized_texts = []
    stop_words = get_stop_words()
    nlp = spacy.load("en_core_web_sm")

    for text in tqdm(docs_as_words):
        lemmatized_tokens = []
        tokens = nlp(" ".join(text))
        for token in tokens:
            # Lemmatize, strip stop words
            lemmatized_token = token.lemma_
            if str(token) and str(lemmatized_token) not in stop_words:
                lemmatized_tokens.append(lemmatized_token)

        lemmatized_texts.append(lemmatized_tokens)
    return lemmatized_texts


URL_REGEX = re.compile(
    "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
    re.M)


def sent_to_words(doc: str) -> List[str]:
    # doc = re.sub('\S*@\S*\s?', '', doc)  # remove emails
    doc = re.sub(URL_REGEX, "", doc)  # remove URLs
    doc = re.sub("\s+", " " "", doc)  # remove newline chars
    doc = re.sub("\'", "", doc)  # remove single quotes
    doc = gensim.utils.simple_preprocess(doc, deacc=True)
    return doc


def normalized_notion_url(page_id: str) -> str:
    return f"notion://www.notion.so/{page_id.replace('-', '')}"


def insert(source_str, insert_str, pos):
    return source_str[:pos] + insert_str + source_str[pos:]


def restore_notion_id_hyphens(notion_id: str) -> str:
    notion_id = insert(notion_id, '-', 20)
    notion_id = insert(notion_id, '-', 16)
    notion_id = insert(notion_id, '-', 12)
    notion_id = insert(notion_id, '-', 8)
    return notion_id

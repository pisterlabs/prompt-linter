from pathlib import Path
from bs4 import BeautifulSoup
from gensim import corpora
from gensim.models import LsiModel
from gensim.parsing.preprocessing import preprocess_string
from gensim.models.coherencemodel import CoherenceModel
import re, sys


def load_articles(data_dir):
    reuters = Path(data_dir)
    for path in reuters.glob('*.sgm'):
        with path.open() as sgm_file:
            try:
                contents = sgm_file.read()
                soup = BeautifulSoup(contents, features="html.parser")
                for article in soup.find_all('body'):
                    yield article.text
            except:
                print("Unexpected error:", sys.exc_info()[0], sys.exc_info()[1])
                yield ""


def load_documents(document_dir):
    print(f'Loading from {document_dir}')
    documents = list(load_articles(document_dir))
    print(f'Loaded {len(documents)} documents')
    return documents


def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


def clean(x):
    x = clean_text(x)
    x = clean_numbers(x)
    return x


def prepare_documents(documents):
    print('Preparing documents')
    documents = [clean(document) for document in documents]
    documents = [preprocess_string(doc) for doc in documents]
    return documents


def create_lsa_model(documents, dictionary, number_of_topics):
    print(f'Creating LSI Model with {number_of_topics} topics')
    document_terms = [dictionary.doc2bow(doc) for doc in documents]
    return LsiModel(document_terms, num_topics=number_of_topics, id2word = dictionary)


def run_lsa_process(documents, number_of_topics=10):
    documents = prepare_documents(documents)
    dictionary = corpora.Dictionary(documents)
    lsa_model = create_lsa_model(documents, dictionary, number_of_topics)
    return documents, dictionary, lsa_model


def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()


def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        documents, dictionary, model = run_lsa_process(articles, number_of_topics=num_topics)
        coherence = calculate_coherence_score(documents, dictionary, model)
        print(f'\nCoherence score for {num_topics} topics: {coherence}')
        yield coherence


document_dir ='/Users/oliver/dev/github/Natural-Language-Processing-Fundamentals/Lesson5/data/reuters'
articles = list(load_articles(document_dir))
min_topics, max_topics = 20,25
coherence_scores = list(get_coherence_values(min_topics, max_topics))

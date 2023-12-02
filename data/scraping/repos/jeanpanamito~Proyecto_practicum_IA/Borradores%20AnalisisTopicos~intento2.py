import gensim
import nltk
import pymongo
from charset_normalizer import models
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.test.test_hdpmodel import dictionary
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import re
from pprint import pprint
from gensim import corpora, models
import numpy as np
import tqdm

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Uri de conexión con Credenciales
uri = "mongodb+srv://mate01:mcxDZa9yU8aUaK2O@cluster0tweet-gp.hkqaqos.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Base de Datos y Colección
mongo_db = 'Preprocessing'
mongo_collection = 'tweets'
db = client[mongo_db]

# Obtener los documentos de la colección
datos = list(db[mongo_collection].find().limit(100))
tweets = [d['full_text'] for d in datos]

stop_words = stopwords.words('spanish')
stop_words.extend(['rt'])
stop_words.extend(['q'])


def preprocess(text):
    text = text.lower()
    text = re.sub('@[A-Za-z0-9_]+', '', text)  # remove users
    text = re.sub('[^a-zA-ZáéíóúÁÉÍÓÚñ. \s]', '', text)  # remove special characters
    text = re.sub('https?://\S+', '', text)  # remove url
    text = re.sub('[^\w\s]', '', text)  # Remove punctuation
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Remove leading/trailing spaces
    return text


def remove_stopwords(text):
    words = text.split(' ')
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    filtered_text = ' '.join(filtered_words)
    return filtered_text


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def get_wordnet_pos(token):
    tag = nltk.pos_tag([token])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_text(text):
    text = preprocess(remove_stopwords(text))
    return lemmatize_text(text)


# Aplicar el preprocesamiento a los tweets
tweetDF = pd.DataFrame(datos)
tweetDF['full_text'] = tweetDF['full_text'].map(preprocess_text)

data_words = list(sent_to_words(tweetDF['full_text']))

bigram = Phrases(data_words, min_count=5, threshold=100)
trigram = Phrases(bigram[data_words], threshold=100)

bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

data_words_trigrams = trigram_mod[data_words]

data_lemmatized = [lemmatize_text(" ".join(doc)) for doc in data_words_trigrams]

id2word = corpora.Dictionary(data_lemmatized)

texts = data_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]

tfidf_model = models.TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]

# Ajustar modelo LDA
lda_model = models.LdaModel(corpus=tfidf_corpus, id2word=id2word, num_topics=10)

coherence_model_lda = CoherenceModel(model=tfidf_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)

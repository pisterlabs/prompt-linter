
import nltk
import pymongo
import tweepy
from nltk import WordNetLemmatizer
from pymongo import collection
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora,models
import pandas as pd
from pprint import pprint
import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis


nltk.download('stopwords')

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

datos = db.tweets.find().limit(5)

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

tweetDF = pd.DataFrame(datos)
tweetDF['full_text'] = tweetDF['full_text'].map(preprocess_text)

data_words = list(sent_to_words(tweetDF['full_text']))
#print(data_words[:100])

bigram = Phrases(data_words, min_count=5, threshold=100) # umbral más alto menos frases.
#trigram = Phraseevs(bigram[data_words], threshold=100)

# Forma más rápida de convertir una oración en un trigrama/bigrama
bigram_mod = Phraser(bigram)
#trigram_mod = Phraser(trigram)

data_words_trigrams = bigram_mod[data_words]

data_lemmatized = [lemmatize_text(" ".join(doc)) for doc in data_words_trigrams]

#print(data_lemmatized[:7][:50])

id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1][0][:30])

tfidf_model = models.TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]

#print(tfidf_corpus[:1][0][:30])

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=15,
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
#print('Coherence Score: ', coherence_lda)

# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()

num_topics = 14

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=0.01,
                                           eta=0.9)

# Visualize the topics
pyLDAvis.enable_notebook()

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)




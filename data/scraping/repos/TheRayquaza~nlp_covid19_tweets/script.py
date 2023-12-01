############ I- Imports ############
print("I- Imports")
# Other models
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
# Sklearn
from sklearn.feature_extraction.text import CountVectorizer
# BERTopic 
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
# NLTK for pre-processing
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# Gensim for results
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
# Others
import pandas as pd
import numpy as np
import re
import string


############################################
############ II- Pre-Processing ############
print("II- Pre-Processing")

## A) Reading data
print("\tA) Reading data")

df = pd.read_csv("./datasets/covid19_tweets.csv")
df = df.head(20000)

## B) Initialize NLTK stemmer and stopwords
print("\tB) Initialize NLTK stemmer and stopwords")

stemmer = PorterStemmer()
nltk.download("stopwords")
stop = stopwords.words("english")
nltk.download('punkt')

## C) Preparing dates for graph over time
print("\tC) Preparing dates for graph over time")

df["date"] = pd.to_datetime(df["date"], format="mixed")
df.sort_values(by="date")

documents = df["text"].values
dates = df["date"].values

for i in range(len(dates)) :
    dates[i] = dates[i].astype('datetime64[D]')


## D) Treating text
print("\tD) Treating text")

def remove_emoji(text):
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"u"\U0001F680-\U0001F6FF"u"\U0001F1E0-\U0001F1FF"u"\U00002500-\U00002BEF"u"\U00002702-\U000027B0"u"\U000024C2-\U0001F251""]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_url(text) :
    text = re.sub(r'https?://\S+', '', text)
    return re.sub(r'http?://\S+', '', text)

def remove_tags(text) :
    return re.sub(r'@\w+', '', text)

def remove_whitespace(text) :
    return re.sub(r"\s+", " ", text)

def pre_processing(documents:np.ndarray) :
    L = []
    for text in documents:
        if text != "" and text and text == text:
            # Removing links
            text = remove_url(text)

            # Removing tags
            text = remove_tags(text)

            # Removing emojis
            text = remove_emoji(text)

            # Removing useless whitespaces
            text = remove_whitespace(text)

            # Tokenize the sentence
            tokens = nltk.word_tokenize(text)

            # Stemming
            stemmed_tokens = [stemmer.stem(token) for token in tokens]

            # Stop word removal
            filtered_tokens = [token for token in stemmed_tokens if token.lower() not in stop]

            # Punctuation removal
            filtered_tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in filtered_tokens if token]

            # Remove empty string
            filtered_tokens = [token for token in filtered_tokens if token != '' and len(token) > 2]

            # Join tokens back into a sentence
            text = ' '.join(filtered_tokens)
            L.append(text)
        else :
            L.append("")
    return L

tweets = pre_processing(documents)

##############################################
############ III- Model selection ############
print("III- Model selection")

embedding = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding)
print("\tA) Selecting embedding model : ", embedding)

umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine')
print("\tB) Selecting dimensionality reduction model : UMAP/", umap_model)

hdbscan_model =  HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True, gen_min_span_tree=True)
print("\tC) Selecting clustering model : HDBSCAN/", hdbscan_model)

vectorizer_model = CountVectorizer(encoding="utf-8", stop_words="english")
print("\tD) Selecting topics tokenizer : CountVectorizer/", vectorizer_model)

ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
print("\tE) Selecting weighting model : CounClassTfidfTransformertVectorizer/", ctfidf_model)

# All steps together
topic_model = BERTopic(
  language="english",
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  ctfidf_model=ctfidf_model,
  calculate_probabilities=True,        
  verbose=True,
)

print("===> Applying fit transform on dataset : ", len(tweets), "<===")
topics, probs = topic_model.fit_transform(tweets)
print("=== Saving model ===")
topic_model.save("./models/best_model")
############################################
############ IV - Model Figures ############
print("IV - Saving model figures")

print("\tBarchart saved : barchart.png")
fig = topic_model.visualize_barchart(top_n_topics=10)
fig.write_image("./plot/barchart.png")

print("\tIntertopic saved : intertopic_distance_map.png")
fig = topic_model.visualize_topics()
fig.write_image("./plot/intertopic_distance_map.png")



############################################
############ V - Model Metrics ############

print("V) Evaluating model metrics using topic coherence")

cv = topic_model.vectorizer_model
X = cv.fit_transform(tweets)
doc_tokens = [text.split(" ") for text in tweets]

id2word = Dictionary(doc_tokens)
texts = doc_tokens
corpus = [id2word.doc2bow(text) for text in texts]

topic_words = []
for i in range(len(topic_model.get_topic_freq())-1):
  interim = []
  interim = [t[0] for t in topic_model.get_topic(i)]
  topic_words.append(interim)

coherence_model = CoherenceModel(topics=topic_words, texts=texts, corpus=corpus, dictionary=id2word, coherence='c_v')
coherence_score = coherence_model.get_coherence()
print("\tCoherence score :", coherence_score)

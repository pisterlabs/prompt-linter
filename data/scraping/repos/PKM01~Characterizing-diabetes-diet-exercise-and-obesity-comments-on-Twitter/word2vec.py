import pandas as pd
import numpy as np
import gensim
import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained Spacy model for preprocessing
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Load tweet dataset
tweets_df = pd.read_csv('new_data.csv')

# Preprocess the tweet text using Spacy
def preprocess_text(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]

tweets_df['processed_text'] = tweets_df['Text'].apply(preprocess_text)

# Train Word2Vec model to obtain word embeddings
w2v_model = Word2Vec(tweets_df['processed_text'], size=100, window=5, min_count=1)

# Create similarity matrix based on word embeddings
similarity_matrix = w2v_model.wv.similarity_matrix(Dictionary(tweets_df['processed_text']).token2id, len(w2v_model.wv.vocab))

# Define function to compute topic coherence using Word Embeddings
def topic_coherence_w2v(topics, similarity_matrix):
    coherence = 0
    topic_terms = [list(topic) for topic in topics]
    for i, topic in enumerate(topic_terms):
        topic_similarity = cosine_similarity([w2v_model[token] for token in topic])
        coherence += np.sum(np.triu(topic_similarity, k=1)) / (len(topic) * (len(topic)-1) / 2)
    return coherence / len(topic_terms)

# Train LDA model and evaluate coherence score using Word Embeddings
doc_term_matrix = [Dictionary(doc) for doc in tweets_df['processed_text']]
for i, doc in enumerate(tweets_df['processed_text']):
    for j, token in enumerate(doc):
        if token in w2v_model.wv.vocab:
            doc_term_matrix[i][j] = doc_term_matrix[i].token2id[token] + len(doc_term_matrix)
            doc_term_matrix.append(list(w2v_model[token]))

lda_model = LdaModel(doc_term_matrix, num_topics=5, id2word=Dictionary(tweets_df['processed_text']))
topics = lda_model.show_topics(num_topics=-1, num_words=10)
coherence_lda_w2v = topic_coherence_w2v(topics, similarity_matrix)

print('Coherence Score (LDA with Word Embeddings):', coherence_lda_w2v)
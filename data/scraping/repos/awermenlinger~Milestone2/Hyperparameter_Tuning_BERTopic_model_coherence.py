## Still hits error wall after trying the latest pull request code

from bertopic import BERTopic
import pandas as pd
import pickle
from random import sample
from umap import UMAP
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import torch

bertopic_file = "models/BERTopic_trained_model.sav"
model = BERTopic.load(bertopic_file)

data_files = ["data/pubmed_articles_cancer_01_smaller.csv", "data/pubmed_articles_cancer_02_smaller.csv",
                "data/pubmed_articles_cancer_03_smaller.csv","data/pubmed_articles_cancer_04_smaller.csv"]

input_data = pd.DataFrame()
print ("loading the files")
for file in data_files:
    df_load = pd.read_csv(file,skip_blank_lines=True)
    input_data = input_data.append(df_load)

input_data.abstract = input_data.abstract.astype('str')
doc_processed = input_data['abstract']

# print ("training the model")
# umap_model = UMAP(n_neighbors=200, n_components=10, min_dist=0.0, metric='cosine')
# topic_model = BERTopic(language="english", calculate_probabilities=False, n_gram_range=(1,1), top_n_words=15,
#                  verbose=True, min_topic_size=300, umap_model=umap_model)
topics = model.get_topics()

print ("preprocess for coherence...")
# Preprocess Documents
documents = pd.DataFrame({"Document": doc_processed,
                          "ID": range(len(doc_processed)),
                          "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

print ("vectorizer and analyzer for coherence...")
# Extract vectorizer and analyzer from BERTopic
vectorizer = model.vectorizer_model
analyzer = vectorizer.build_analyzer()

# Extract features for Topic Coherence evaluation
print ("Extract features for Topic Coherence evaluation...")
words = vectorizer.get_feature_names()
tokens = [analyzer(doc) for doc in cleaned_docs]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]
topic_words = [[words for words, _ in model.get_topic(topic)] 
               for topic in range(len(set(topics))-1)]

# Evaluate
print ("Evaluate...")
coherence_model = CoherenceModel(topics=topic_words, 
                                 texts=tokens, 
                                 corpus=corpus,
                                 dictionary=dictionary, 
                                 coherence='c_v')
coherence = coherence_model.get_coherence()



print("BERTopic coherence score: {}".format(coherence))
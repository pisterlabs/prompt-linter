import os
from datetime import datetime
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pandas as pd


# save bertopic model
def save_bertopic_model(model, filename = 'bertopic_', model_dir = "/kaggle/working/model"):
    # retrieve time
    now = datetime.now()
    # create the directory if it doesn't exist
    try:
        os.makedirs(model_dir)
    except:
        pass
    model.save(model_dir + '/' + filename + now.strftime("%d%m%Y_%H%M%S"))


def compute_coherence(topic_model, filtered_text, topics, metric = 'c_v'):
    documents = pd.DataFrame({"Document": filtered_text,
                            "ID": range(len(filtered_text)),
                            "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    #words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
                for topic in range(len(set(topics))-1)]

    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence='c_v')
    return coherence_model.get_coherence()


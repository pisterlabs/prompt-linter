## topic evaluation 

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
import pandas as pd
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel


def prepare_docs_for_coherence_eval(docs,topics,probabilities,model):
    documents = pd.DataFrame({"Document": docs,
                          "ID": range(len(docs)),
                          "Topic": topics,
                          "Topic_prob": probabilities})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    #print(documents_per_topic.head())
    # Extract vectorizer and analyzer from BERTopic
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)
    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in model.get_topic(topic) if words!=''] 
                for topic in range(len(set(topics))-1)]
    topic_words = [t for t in topic_words if len(t) >0] ## for some reason some topics has all "" as topic words

    return topic_words,tokens,corpus,dictionary

def get_coherence_score(topic_words,tokens,corpus,dictionary,n_workers=-1):
    # Evaluate
    # print("n_workers: {}".format(n_workers))
    coherence_model = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence='c_v',
                                    processes=n_workers)
    coherence = coherence_model.get_coherence()
    
    return coherence

def eval_coherence_score(docs,topics,probabilities,model,n_workers=-1):
    topic_words,tokens,corpus,dictionary = prepare_docs_for_coherence_eval(docs,topics,probabilities,model)
    coherence = get_coherence_score(topic_words,tokens,corpus,dictionary,n_workers=n_workers)
    
    return coherence
    

def BERTopic2OCTIS_output(topic_model):
    topics_rep = topic_model.get_topics()
    OCTIS_topics = [[k[0] for k in top] for top in list(topics_rep.values())]
    return {'topics':OCTIS_topics}

def eval_diversity_score(topic_model):
    
    diversity_metric = TopicDiversity(topk=topic_model.top_n_words)
    octis_model_output = BERTopic2OCTIS_output(topic_model)
    topic_diversity_score = diversity_metric.score(octis_model_output) # Compute score of the metric

    return topic_diversity_score

if __name__ == "__main__":
    pass
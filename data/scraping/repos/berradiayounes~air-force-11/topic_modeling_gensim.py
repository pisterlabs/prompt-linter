import re
import numpy as np

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Enable logging for gensim
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR
)


# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip


def compute_coherence_values(
    dictionary, corpus, texts, limit, start=2, step=3
):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    mallet_path = "model/mallet-2.0.8/bin/mallet"
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit + 1, step):
        model = gensim.models.wrappers.LdaMallet(
            mallet_path,
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def topic_modeling(preprocessed_data, start, limit, step, num_words_per_topic):
    """
    Perform topic modeling using Gensim, find the optimal number of topics
    and the associated keywords.

    Parameters:
    ----------
    preprocessed_data : Reviews dataframe after preprocessing step
    start : Minimum number of topics considered
    limit : Maximum number of topics considered
    step : Increment during the search of the optimal number of topics
    num_words_per_topic : Number of words to display per topic

    Returns:
    -------
    topic_keywords : Dictionary with the topics and the associated keywords
    """
    # Create Dictionary
    id2word = corpora.Dictionary(preprocessed_data)
    # Create Corpus
    texts = preprocessed_data
    # Term Document Frequency ("doc2bow" converts document into the bag-of-words
    # (BoW) format = list of (token_id, token_count) tuples.)
    corpus = [id2word.doc2bow(text) for text in texts]

    # Find optimal number of topics and associated LDA model
    model_list, coherence_values = compute_coherence_values(
        dictionary=id2word,
        corpus=corpus,
        texts=preprocessed_data,
        start=start,
        limit=limit,
        step=step,
    )

    optimal_model = model_list[np.argmax(coherence_values)]
    model_topics = optimal_model.show_topics(
        formatted=False, num_words=num_words_per_topic
    )

    topic_keywords = {}
    for topic_number in range(len(model_topics)):
        topic = model_topics[topic_number][0]
        keywords = [element[0] for element in model_topics[topic_number][1]]
        topic_keywords[topic] = keywords

    return topic_keywords

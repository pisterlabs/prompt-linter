import gensim
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from typing import List


def prepare_input(data: List[List[str]]):
    """
    Defines bigram, dictionary and corpus for the topic model input.
    Args:
        data (List[List[str]]): The dataset containing lemmas.
    """

    bigram = gensim.models.Phrases(data)
    dictionary = Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]

    return bigram, dictionary, corpus


def compute_c_v(
    dictionary: List[tuple],
    corpus,
    texts: List[List[str]],
    min_topics: int,
    max_topics: int,
    step: int,
) -> List[float]:
    """
    Computes c_v coherence score.
    Args:
       dictionary (List[tuple]): A list of tuples (word_id, word_frequency).
       corpus: A document-term matrix.
       texts (List[List[str]]): A corpus with lemmatized strings (tokens).
       min_topics (int): The minimum number of topics to identify.
       max_topics (int): The maximum number of topics to identify.
       step (int): Step with which to iterete across the topics.
    Returns:
        List[float]: The list with c_v scores for each number of topics.
    """

    coherence_values = []

    for num_topics in range(min_topics, max_topics, step):

        model = LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            update_every=1,
            passes=10,
            per_word_topics=True,
        )

        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )

        coherence_values.append(coherencemodel.get_coherence())

    return coherence_values

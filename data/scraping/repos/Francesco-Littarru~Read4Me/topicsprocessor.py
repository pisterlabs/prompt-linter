"""
**What is this module for?**

This module provides a class to further process topics generated from a
gensim's :mod:`~gensim.models.hdpmodel`.

Probabilistic topic models create topics that are not always easily interpretable,
that is, they can generate topics with words not intuitively related to each other.
The inferred topics may contain unrelated words that need to be removed or weighed down.
Only selecting the first words from a topic may not suffice.

In this module the topics are filtered, the expected result is to have shorter
topics with higher internal coherence from a human standpoint.

**What classes are there?**

* :class:`TopicsProcessor`

**How can you use it?**

.. note::
    To obtain the topics in the right format from the :mod:`~gensim.models.hdpmodel` you should use the method
    :meth:`~gensim.models.hdpmodel.HdpModel.show_topics` with the parameter "formatted" set to False.

Initialize an instance with the topics to process, the reference corpus and dictionary used to train the topic model.

.. code-block::

    tp = TopicsProcessor(topics, corpus, dictionary)

Filter the topics and save the result to a pickle file.

.. code-block::

    tp.filter_topics()
    tp.save_topics(pathlib.Path("filtered_topics.pkl"))

For the specifics of filtering and its parameters see the documentation below.

.. todo::
    Intra-topic clustering with semantic vectors for better filtering --- might be slow.
"""

from collections import defaultdict
import logging
import math
from pathlib import Path
import pickle

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy


logging.basicConfig(format="%(levelname)s %(asctime)s %(filename)s %(funcName)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("datafactory")
logger.setLevel(logging.INFO)


class TopicsProcessor:
    """
    This class does the post-processing of the topics obtained through a Gensim topic model.
    The goal is to modify the original statistically inferred topics to ones with a stronger semantical coherence from
    the point of view of a human reader.
    """
    def __init__(self,
                 topics: list[tuple[int, list[tuple[str, float]]]],
                 corpus: list[list[str]],
                 dictionary: Dictionary,
                 min_dict_count: float = 20,
                 min_topic_proba: float = 0.005,
                 min_topic_coherence: float = 0.4,
                 n_process: int = -1):
        """
        Initialize an instance for processing the topics.

        :param topics: Not formatted topics as given by a gensim model.
        :param corpus: A reference corpus, used to compute coherence on topics.
        :param dictionary: Gensim :class:`~gensim.corpora.dictionary.Dictionary` for topics and corpus.
        :param min_dict_count: Threshold for filtering by reconstructed word counts.
        :param min_topic_proba: Discard topics with lower probability. Topic probability as sum of word probabilities.
        :param min_topic_coherence: Keep topics with coherence of at least this value.
        :param n_process: Number of processes to use.
        """
        self.__model_topics = topics
        self.__corpus = corpus
        self.__dictionary = dictionary
        self.__min_dict_count = min_dict_count
        self.__min_proba = min_topic_proba
        self.__topic_coherence = min_topic_coherence
        self.__num_topics = len(self.__model_topics)          # infer number of topics
        self.__num_words = len(self.__model_topics[0][1])     # infer number of words per topic
        self.__word_topic_map = defaultdict(list)             # {word: [(topic_id, weight), ...], ...}
        self.__topic_proba: list[tuple[int, float]] = []
        self.__n_process = n_process

    def filter_topics(self):
        """
        Perform filtering on the topics based on the parameters set up during initialization.

        Filters:

        #. Remove from the topics all the words that have a reconstructed word count less than min_dict_count.
        #. Remove all the topics with remaining probability mass less than min_topic_proba.
        #. Remove the topics with a coherence value less than min_topic_coherence.

        See :class:`TopicsProcessor` for "`min_*`" parameters.

        .. note::
            The reconstructed word count is computed as the word probability in the topic
            multiplied by its count in the dictionary.


        After filtering, rescale the weights of the words such that the sum of the inter-topic probabilities of
        a given word is one.

        Example: If a word appears only in two topics and in topic 1 has weight 0.03 and in topic 6 has weight 0.01,
        then after rescaling, the weights become 0.75 in topic 1 and 0.25 in topic 6.

        Rescaling allows to compute the tfidf on the topics as if they were documents, the term count of
        a word in a topic is thus given by its count in the dictionary multiplied by its weight.

        Finally, apply tfidf using the counts from the dictionary and the word topic weights.
        """
        self.__filter_by_reconstructed_word_counts(min_count=self.__min_dict_count)
        self.__filter_by_topic_proba(min_proba=self.__min_proba)
        self.__filter_by_coherence(threshold=self.__topic_coherence)
        self.__per_word_rescale()
        self.__apply_tfidf()

    def __filter_by_reconstructed_word_counts(self, min_count: float):
        """
        Discard words with reconstructed word count lower than min_count.
        The reconstructed word count is computed by multiplying the word probability in the topic with
        the word count in the dictionary.

        :param min_count: Threshold value.
        """
        topics = []
        _deduplication_sets = []
        logger.info("Filtering by reconstructed counts.")
        for (index, topic) in self.__model_topics:
            filtered = [(word, weight) for (word, weight) in topic
                        if self.__word_count(word)*weight >= min_count]
            if len(filtered) > 0:
                _words = {w for (w, _) in filtered}
                if _words not in _deduplication_sets:
                    topics.append((index, filtered))
                    _deduplication_sets.append(_words)
        self.__model_topics = topics
        logger.info("Done.")

    def __filter_by_topic_proba(self, min_proba: float):
        """
        Discard topics with a probability mass lower than min_proba.
        The probability mass of a topic is computed as the sum of the probabilities of its words.
        If this method is used as the first stage of filtering, then the probability mass of each topic sums to 1
        and thus this filter has no effect.

        :param min_proba: Minimum probability threshold.
        """
        logger.info("Filtering by topic probability mass.")
        probs = TopicsProcessor.topic_probabilities(self.__model_topics)
        topics = [(t_id, topic) for (t_id, topic) in self.__model_topics if probs[t_id] >= min_proba]
        logger.info("Done.")
        self.__model_topics = topics

    def __filter_by_coherence(self, threshold: float = 0.4):
        """
        Compute the coherence for each topic, discard topics with coherence value below the threshold.

        :param threshold: Coherence value, float between 0 and 1.
        """
        logger.info("Filtering by topic coherence.")
        topics_as_words = [[word for (word, value) in topic] for (_, topic) in self.__model_topics]
        cm = CoherenceModel(topics=topics_as_words,
                            texts=self.__corpus,
                            dictionary=self.__dictionary,
                            coherence='c_v',
                            processes=self.__n_process)
        logger.info("Computing topic coherences.")
        values = cm.get_coherence_per_topic()
        values = zip([t_id for (t_id, _) in self.__model_topics], values)
        co_ids = {t_id for (t_id, v) in values if v >= threshold}
        topics = [(t_id, topic) for (t_id, topic) in self.__model_topics if t_id in co_ids]
        logger.info("Done.")
        self.__model_topics = topics

    def __per_word_rescale(self):
        """
        Rescale the weights of the words such that the sum of the inter-topic probabilities of a given word is one.
        Example:
        If the word "storm" appears (only) in topic 2, 6 and 80 with different weights,
        rescale their weights to sum to one.
        """
        logger.info("Rescaling word weights.")
        wtm = self.__map_words_to_topics()
        probs = TopicsProcessor.topic_probabilities(self.__model_topics)
        topic_recount = defaultdict(list)
        for word in wtm.keys():
            weights = numpy.asarray([w for (_, w) in wtm[word]])
            topic_ids = [t_id for (t_id, _) in wtm[word]]
            weights = weights/sum(weights)
            wtm[word] = list(zip(topic_ids, weights))
            for (t_id, w) in wtm[word]:
                topic_recount[t_id].append((word, w*probs[t_id]))
        self.__model_topics = [(t_id, sorted(topic_recount[t_id], key=lambda x: x[1], reverse=True))
                               for t_id in sorted(topic_recount.keys())]
        logger.info("Done.")

    def __apply_tfidf(self):
        """
        Use the topics as if they were documents and apply tfidf on their words.
        Per-topic word counts are computed multiplying the count in the dictionary by the weight
        they have in the topic.
        """
        wtm = self.__map_words_to_topics()
        n_docs = len(self.__model_topics)
        for t_count, (t_id, topic) in enumerate(self.__model_topics):
            words, values = zip(*topic)
            tfidf_w = []    # store tfidf weights
            for w_pos in range(len(words)):
                tf = math.log10(self.__word_count(words[w_pos]) * values[w_pos] + 1)
                idf = math.log10(n_docs/len(wtm[words[w_pos]]))
                tfidf_w.append(tf*idf)
            self.__model_topics[t_count] = (t_id, sorted(list(zip(words, tfidf_w)), key=lambda x: x[1], reverse=True))

    def __map_words_to_topics(self) -> dict[str, list[tuple[int, float]]]:
        """
        Mapping of words to topics.
        Map the words in the topics to a list of tuples of the form <(int) topic, (float) weight>,
        representing occurrence in the topic with the assigned weight.
        Example: {"dog": [(0, 0.6), (2, 0.3), (18, 0.1)]}.

        :return: Dictionary.
        """
        for topic_id, topic in self.__model_topics:
            for (word, weight) in topic:
                self.__word_topic_map[word].append((topic_id, weight))
        return self.__word_topic_map

    @staticmethod
    def topic_probabilities(topics: list[tuple[int, list[tuple[str, float]]]]) -> dict[int, float]:
        """
        Compute the probability mass of each topic.
        For each topic, sum the probabilities of the words to obtain its probability.

        :param topics: List of topics.
        :return: Dict with topics as keys and topic probabilities as values.
        """
        values = [sum([e[1] for e in topic]) for (_, topic) in topics]
        probs = numpy.asarray(values)/sum(values)
        probabilities: defaultdict[int, float] = defaultdict(float)
        for index, (t_id, topic) in enumerate(topics):
            probabilities[t_id] = probs[index]
        logger.info("Topic probabilities computed.")
        return probabilities

    def __word_count(self, word: str) -> int:
        """
        Get the count of a word given as a string.

        :param word: Word as a string.
        :return: Count for the word in the Gensim dictionary.
        """
        return self.__dictionary.dfs[self.__dictionary.token2id[word]]

    def save_topics(self, path: Path):
        """
        Save the processed topics to a pickle file.

        :param path: Path to file.
        """
        with path.open('wb') as file:
            pickle.dump(self.__model_topics, file)
        logger.info(f"Topic instance saved to {path}.")


if __name__ == '__main__':
    pass

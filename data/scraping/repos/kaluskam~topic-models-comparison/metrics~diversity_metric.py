from utils.data_structures import *
from metrics.metric import Metric
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO, LogOddsRatio, \
    WordEmbeddingsInvertedRBO, WordEmbeddingsInvertedRBOCentroid, KLDivergence
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import re


class TopicDiversityMetric(Metric):
    """A class to calculate Topic Diversity metric"""

    def __init__(self, flag=False, range=(0, 1), parameters=None):
        """
        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
                topk (int) -  top k words on which the topic diversity will be computed
        """
        self.name = "Topic Diversity"
        self.description = "Topic Diveristy calculates the ratio of diverse words used to describe the topics."
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        """
        Retrieve the score of the metric

        Returns
        -------
        float : topic diversity
        """
        super().evaluate(inputData, outputData)
        td = TopicDiversity(**self.parameters["diversitymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return td.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"diversitymodel": {}}


class InvertedRBOMetric(Metric):
    """A class to calculate Inverted Ranked Biased Overlap metric"""

    def __init__(self, flag=False, range=(0, 1), parameters=None):
        """
        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
                topk (int) :  top k words on which the topic diversity will be computed
                weight (float) :  weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns to
                average overlap. (default 0.9)
        """
        super().__init__(flag, range, parameters)
        self.name = "Inverted RBO"
        self.description = "Metric calculates average diversity of topic-word lists using Inverted Ranked Biased Overlap " \
                           "- a method to compare two ranked lists."
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        """
        Retrieve the score of the metric

        Returns
        -------
        float : Inverted Ranked-Biased Overlap calculated on word lists inside topics
        """
        super().evaluate(inputData, outputData)
        invrbo = InvertedRBO(**self.parameters["diversitymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return invrbo.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"diversitymodel": {}}


class LogOddsRatioMetric(Metric):
    """A class to calculate Log Odds Ratio metric"""

    def __init__(self, flag=False, range=(0, float("inf"))):
        self.name = "Log Odds Ratio"
        self.description = "Log Odds Ratio compares the usage of a word across " \
                           "different documents. The metric is calculated based on a topic-word probability matrix."
        super().__init__(flag, range)

    def evaluate(self, inputData, outputData):
        """
        Retrieve the score of the metric

        Returns
        -------
        float : Log Odds Ratio calculated on topic-word probability matrix
        """
        super().evaluate(inputData, outputData)
        log_odds_ratio = LogOddsRatio()
        topics_word_dict = {"topic-word-matrix": outputData.topic_word_matrix}
        return log_odds_ratio.score(topics_word_dict)


class WordEmbeddingsInvertedRBOMetric(Metric):
    """A class to calculate Word Embedding Inverted Ranked Biased Overlap metric"""

    def __init__(self, flag=False, range=(0, 1), parameters=None):
        """
        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
                topk (int): top k words on which the topic diversity will be computed
                word2vec_path (str): word embedding space in gensim word2vec format
                weight (float): weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns average overlap. (Default 0.9)
                normalize (bool): if true, normalize the cosine similarity
                binary (bool): True if the word2vec file is binary, False otherwise (default True)
        """
        self.name = "Word Embeddings Inverted RBO"
        self.description = "Metric calculates average pairwise diversity of topic-word vector lists using Inverted Ranked Biased Overlap and embedding model (the default embedding model is word2vec-google-news-300)."
        super().__init__(flag, range, parameters)
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        """
        Retrieve the score of the metric

        Returns
        -------
        float : Word Embedding Inverted Ranked Biased Overlap calculated on words from topics
        """
        super().evaluate(inputData, outputData)
        we_inv_rbo = WordEmbeddingsInvertedRBO(**self.parameters["diversitymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return we_inv_rbo.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"diversitymodel": {}}


class WordEmbeddingsInvertedRBOCentroidMetric(Metric):
    """A class to calculate Word Embedding Inverted Ranked Biased Overlap Centroid metric"""

    def __init__(self, flag=False, range=(0, 1), parameters=None):
        """
        Parameters
        ----------
        flag : bool
            indicates whether the higher or lower score is better
        range: tuple
            minimum and maximum value of a metric
        parameters: dict, optional
            dictionary with keys:
                topk (int): top k words on which the topic diversity will be computed
                word2vec_path (str): word embedding space in gensim word2vec format
                weight (float): Weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns average overlap. (Default 0.9)
                normalize (bool): if true, normalize the cosine similarity
                binary (bool): True if the word2vec file is binary, False otherwise (default True)
        """
        super().__init__(flag, range, parameters)
        self.name = "Word Embeddings Inverted RBO Centroid"
        self.description = "Metric calculates average diversity of topic-word vector lists using Inverted Ranked Biased Overlap and embedding model (the default embedding model is word2vec-google-news-300). " \
                           "The diversity is calculated between each word vector list and mean of word vectors for each topic."
        if parameters is None:
            self.init_default_parameters()

    def evaluate(self, inputData, outputData):
        """
        Retrieve the score of the metric

        Returns
        -------
        float : Word Embedding Inverted Ranked Biased Overlap Centroid calculated on words from topics
        """
        super().evaluate(inputData, outputData)
        we_inv_rbo = WordEmbeddingsInvertedRBOCentroid(**self.parameters["diversitymodel"])
        topics_dict = {"topics": outputData.get_topics()}
        return we_inv_rbo.score(topics_dict)

    def init_default_parameters(self):
        self.parameters = {"diversitymodel": {}}

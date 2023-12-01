import pandas as pd
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


class TopicEval:
    """This is a class for evaluating topic models.
    It returns a dictionary with the evaluation metrics.

    Attributes:
        vocab (dict): The vocabulary of the topic model.
        text (list): The list of documents used as reference for the evaluation.

    Methods:
        topic_coherence(self, metric: str = "c_v", topics: list): Returns the topic coherence of the topics (using CoherenceModel).
        topic_diversity(self, topics: list): Returns the topic diversity of the topics.
        semantic_purity(self, topics: list): Returns the syntax composition (stop words and punctuations) of the topics.
    """

    def __init__(self,
                 vocab: dict,
                 text: list) -> None:
        # vocab is a dictionary with the words as keys and the indices as values
        # converting the vocab to pandas dataframe with index and word columns
        vocab = {v: k for k, v in vocab.items()}
        self.vocab = pd.DataFrame.from_dict(vocab, orient='index', columns=['word'])
        self.vocab['index'] = self.vocab.index
        self.text = text
        self.dictionary = Dictionary(self.text)

    def topic_coherence(self, metric: str, topics: list) -> float:
        cm = CoherenceModel(topics=topics,
                            texts=self.text,
                            dictionary=self.dictionary,
                            coherence=metric)
        return cm.get_coherence()

    def topic_diversity(self, topics: list) -> float:
        unique_words = set()
        for topic in topics:
            unique_words.update(topic)
        return len(unique_words) / (len(topics) * len(topics[0]))

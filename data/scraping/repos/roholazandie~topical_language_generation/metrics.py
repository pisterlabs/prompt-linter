from collections import Counter

import spacy
from gensim.models import CoherenceModel
import numpy as np
from configs import LDAConfig, LSIConfig
from lda_model import LDAModel
from gensim.topic_coherence import segmentation
import logging

from lsi_model import LSIModel

logger = logging.getLogger(__name__)

class Distinct:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    # def distinct_1(self, text):
    #     tokens = [token.text for token in self.nlp(text)]
    #     one_grams = set(tokens)
    #     dist_1 = len(one_grams)/len(tokens)
    #     return dist_1
    #
    # def distinct_2(self, text):
    #     tokens = [token.text for token in self.nlp(text)]
    #     bigrams = set(zip(*[tokens[i:] for i in range(2)]))
    #     dist2 = len(bigrams)/(len(tokens)-1)
    #     return dist2
    #
    # def distinct_3(self, text):
    #     tokens = [token.text for token in self.nlp(text)]
    #     trigrams = set(zip(*[tokens[i:] for i in range(3)]))
    #     dist3 = len(trigrams)/(len(tokens)-2)
    #     return dist3



    def distinct_n(self, example, n):
        counter = Counter()
        n_total = 0
        n_distinct = 0
        example = [token.text for token in self.nlp(example)]
        for token in zip(*(example[i:] for i in range(n))):
            if token not in counter:
                n_distinct += 1
            elif counter[token] == 1:
                n_distinct -= 1
            counter[token] += 1
            n_total += 1

        return float(n_distinct/n_total)

class TopicCoherence:

    def __init__(self, config):
        if type(config) == LDAConfig:
            self.model = LDAModel(config)
        elif type(config) == LSIConfig:
            self.model = LSIModel(config)

        # config_file = "../configs/alexa_lsi_config.json"
        # config = LSIConfig.from_json_file(config_file)
        # model = LSIModel(config=config, build=False)

        self.dictionary = self.model.get_dictionary()
        temp = self.dictionary[0]  # This is only to "load" the dictionary.
        self.cm = CoherenceModel(model=self.model.get_model(),
                            texts=self.model.get_docs(),
                            dictionary=self.dictionary,
                            coherence="c_w2v")

    def get_coherence(self, doc):
        tokens = self.model.tokenizer.tokenize(doc)
        # todo try truncated version of doc_ids, those that are in truncated dictionary
        doc_ids = []
        for token in tokens:
            try:
                tid = self.dictionary.token2id[token]
            except:
                logging.debug("Unknown token: " + str(token))
                continue
            doc_ids.append(tid)

        doc_ids = [np.array(doc_ids)]
        #doc_ids = [np.array([self.dictionary.token2id[token] for token in tokens])]

        segmented_doc = segmentation.s_one_set(doc_ids)

        doc_coherence = self.cm.get_coherence_per_topic(segmented_doc)[0]
        return doc_coherence


if __name__ == "__main__":
    metrics = Distinct()
    #text = "Yes, you will get distinct words (though punctuation will affect all of this to a degree). To generate sentences, I assume that you want something like a Markov chain? I actually wrote up an article on word generation using markov chains a few years ago. The basic ideas are the same: ohthehugemanatee.net/2009/10/…. You'll need to find a way to label starting words in the data structure which this does not do, as well as ending or terminal words. "
    text = "this is a very simple example example example"
    dist1 = metrics.distinct_1(text)
    dist2 = metrics.distinct_2(text)
    dist3 = metrics.distinct_3(text)
    print(dist1)
    print(dist2)
    print(dist3)

    print("distinct", metrics.distinct_n(text, 2))

    # config_file = "../configs/alexa_lda_config.json"
    # config = LDAConfig.from_json_file(config_file)
    # topic_coherence = TopicCoherence(config)
    # doc = "text text text"
    # coherence = topic_coherence.get_coherence(doc)
    # print(coherence)
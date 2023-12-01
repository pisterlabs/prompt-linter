import json
from aides_dataset import AidesDataset
import gensim
import gensim.corpora as corpora
import os
import argparse
from sklearn.model_selection import train_test_split
from utils import create_logger
import matplotlib
from gensim.models import CoherenceModel
# import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import pickle
import math
import pandas as pd
import numpy as np
# DEBUG
from pdb import set_trace as bp


def count_word_occurence_in_topics(topics, occ_thr=3):
    all_words = [list(v.keys())[0] for val in topics.values() for v in val]
    count_occurences = {k: all_words.count(k) for k in all_words}
    sorted_tuples = sorted(count_occurences.items(), key=lambda item: item[1])
    sorted_count_occurences = {k: v for k, v in sorted_tuples}
    frequent_words = {k: v for k, v in sorted_count_occurences.items() if v >= occ_thr}
    return sorted_count_occurences, frequent_words


def split_topic_words(words):
    words_probs = words.split('+')
    words_probs = [string.split('*') for string in words_probs]
    words_probs = [{element[1]: float(element[0])} for element in words_probs]
    return words_probs


def create_topic_dictionnary(topics):
    dict_topics = dict.fromkeys(range(len(topics)))
    for topic in topics:
        key, words = topic
        words_probs = split_topic_words(words)
        dict_topics[key] = words_probs
    return dict_topics


def read_df_hparams(df_hparams):
    list_hparams = []
    for i in range(len(df_hparams)):
        hparams = df_hparams.iloc[i].to_dict()
        for key, val in hparams.items():
            if pd.isna(val):
                hparams[key] = None
        list_hparams.append(hparams)
    return list_hparams


def merge_results(results):
    df_results = []
    for result in results:
        df_results.append(pd.DataFrame.from_dict(result))
    return df_results


class LDATopicModel:
    def __init__(self, dataset, hyper_params, out_path="output/lda_topic_model"):
        self.num_topics = hyper_params.num_topics
        self.dataset = dataset
        self.hyper_params = hyper_params
        self.out_path = self.create_output_path(out_path)
        _ = self.get_corpus()
        self.init_results()


    def create_output_path(self, out_path):
        """create the output path for saving LDA topic model results"""
        out_path = os.path.join(out_path, '{}w_{}r_{}topics_{}bs_{}it_eta-{}_alpha-{}_offset{}_decay{}'.format(self.hyper_params.words_num,
                                                                                                       self.hyper_params.words_ratio,
                                                                                                       self.num_topics,
                                                                                                        self.hyper_params.update_every,
                                                                                                       self.hyper_params.iterations,
                                                                                                       self.hyper_params.eta,
                                                                                                       self.hyper_params.alpha,
                                                                                                       self.hyper_params.offset,
                                                                                                       self.hyper_params.decay))
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        self.logger = create_logger(os.path.join(out_path, 'eval_log.log'))
        return out_path

    def init_results(self):
        name_metrics = ["train_perplexity", "test_perplexity", "train_coherence", "test_coherence", "frequent_words",
                        "topics", "vocab_size"]
        self.results = dict.fromkeys(name_metrics)

    def save_results(self):
        # results_df = pd.DataFrame.from_dict(self.results)
        results_df = pd.DataFrame(self.results.items(), columns=["metric", "value"]).T
        results_df.to_csv(os.path.join(self.out_path, "results.csv"), header=False)

    def get_corpus(self):
        """get the corpus given the bag-of-words for each description"""
        processed_data = self.dataset.get_data_words()
        data_train, data_test = train_test_split(processed_data, test_size=100, random_state=123)
        data_words = processed_data.values.flatten()
        train_data_words = data_train.values.flatten()
        test_data_words = data_test.values.flatten()
        id2word = corpora.Dictionary(data_words)  # the vocabulary is built upon all data
        # Create train Corpus & test corpus
        train_corpus = [id2word.doc2bow(feature_words) for feature_words in train_data_words]
        test_corpus = [id2word.doc2bow(feature_words) for feature_words in test_data_words]
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus
        self.train_data = train_data_words
        self.test_data = test_data_words
        self.id2word = id2word
        # if self.out_path is not None:
        # vocab_out_path = os.path.join(self.out_path, "id2word.json")
        # with open(vocab_out_path, 'w') as f:
        #     json.dump(dict(id2word), f, ensure_ascii=False)
        return data_words

    def train_LDA_model(self):
        """train the LDA model and return the topics"""
        lda_model = gensim.models.LdaModel(corpus=self.train_corpus,
                                           id2word=self.id2word,
                                           num_topics=self.num_topics, alpha=self.hyper_params.alpha, eta=self.hyper_params.eta,
                                           update_every=self.hyper_params.update_every, iterations=self.hyper_params.iterations,
                                           decay=self.hyper_params.decay, offset=self.hyper_params.offset)
        topics = lda_model.print_topics()
        return lda_model, topics

    def compute_eval_metrics(self, lda_model):
        train_ppl = round(lda_model.log_perplexity(self.train_corpus), 2)
        test_ppl = round(lda_model.log_perplexity(self.test_corpus), 2)
        train_coherence_lda = CoherenceModel(model=lda_model, corpus=self.train_corpus, texts=self.train_data,
                                             dictionary=self.id2word, coherence='u_mass')
        train_coherence = round(train_coherence_lda.get_coherence(), 3)
        test_coherence_lda = CoherenceModel(model=lda_model, corpus=self.test_corpus, texts=self.test_data,
                                            dictionary=self.id2word, coherence='u_mass')
        test_coherence = round(test_coherence_lda.get_coherence(), 3)
        self.results["train_perplexity"] = train_ppl
        self.results["test_perplexity"] = test_ppl
        self.results["train_coherence"] = train_coherence
        self.results["test_coherence"] = test_coherence
        self.results["vocab_size"] = len(self.id2word)
        return (train_ppl, train_coherence_lda), (test_ppl, test_coherence_lda)

    def vizualize_topics(self, lda_model):
        pass
        # LDAvis_prepared = gensimvis.prepare(lda_model, self.train_corpus, self.id2word)
        # with open("plots/ldavis_prepared.html", 'wb') as f:
        #     pickle.dump(LDAvis_prepared, f)
        # # load the pre-prepared pyLDAvis data from disk
        # with open("plots/ldavis_prepared.html", 'rb') atrain_coherence = train_coherence_lda.get_coherence()s f:
        #     LDAvis_prepared = pickle.load(f)
        # pyLDAvis.save_html(LDAvis_prepared, "plots/ldavis_prepared.html")

    def postprocess_topics(self, lda_model, topics, num_descr=7):
        "look at word occurences in topics and in a txt file a sample of descriptions & their topic."
        topics = create_topic_dictionnary(topics)
        self.results["topics"] = str(topics)
        self.logger.info(
            '-' * 40 + "TOPICS" + '-' * 40)
        for key, val in topics.items():
            self.logger.info('{}:{}'.format(key, val))
        self.logger.info('-' * 60)
        words_occurences, frequent_occurences = count_word_occurence_in_topics(topics)
        self.logger.info(
            "-----------------------------frequent words occurrence in topics-----------------------------------------------------")
        self.logger.info("number of frequent words: {}".format(len(frequent_occurences)))
        self.logger.info(frequent_occurences)
        self.results["frequent_words"] = str(frequent_occurences)
        for descr_id in range(num_descr):
            descr, descr_topic = self.get_topic_per_description(descr_id, lda_model, topics)
            self.logger.info("DESCRIPTION:")
            self.logger.info(str(list(descr.values())[0]))
            self.logger.info("TOPICS:")
            for topic in descr_topic:
                self.logger.info(topic)
            self.logger.info('-' * 60)

    def get_topic_per_description(self, description_id, lda_model, topics):
        """Get the most likely topics for the description id with their percentage."""
        topics_rate = lda_model[self.train_corpus[description_id]]
        topic_ids = [t[0] for t in topics_rate]
        topic_prop = [t[1] for t in topics_rate]
        description_topics = [({k: v}, topics[k]) for k, v in zip(topic_ids, topic_prop)]
        description = {self.dataset.aides[description_id]["id"]: self.dataset.aides[description_id]["description"]}
        return description, description_topics


if __name__ == '__main__':
    from argparse import Namespace

    # load hyper-parameters
    csv_hparams = "data/csv_model_hparams.csv"
    df_hparams = pd.read_csv(csv_hparams)
    # examples of hparams:
    # hparams = {'words_num': 500, 'words_ratio': 0., 'num_topics': 5, 'update_every': 1, 'iterations': 50,
    #                 'alpha': 'symmetric', 'eta': None, 'offset': 1, 'decay': 0.5}
    # alpha is either 'symmetric' or 'assymmetric'
    # eta is either None or 'auto'
    # decay is between 0.5 & 1.eta

    list_hparams = read_df_hparams(df_hparams)
    list_results = []

    for hparams in list_hparams:
        hparams = Namespace(**hparams)
        print('Hyper-parameters:', hparams)
        # Get the aides Dataset
        aides_dataset = AidesDataset("data/AT_aides_full.json", words_num=hparams.words_num, words_ratio=hparams.words_ratio)
        # Build the LDA Topic Model
        lda_TM = LDATopicModel(dataset=aides_dataset, hyper_params=hparams)
        # Train the LDA Model
        lda_model, topics = lda_TM.train_LDA_model()
        lda_TM.postprocess_topics(lda_model=lda_model, topics=topics)
        lda_TM.compute_eval_metrics(lda_model=lda_model)
        lda_TM.save_results()
        list_results.append(lda_TM.results)

        # for unseen document, we can use:
        # get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)

        print("done")

    df_results = pd.DataFrame.from_records(list_results)
    df_results = pd.concat([df_hparams, df_results], axis=1)
    df_results.to_csv(os.path.join("output/lda_topic_model", "results_hparams.csv"))
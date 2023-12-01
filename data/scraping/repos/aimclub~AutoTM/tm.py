import glob
import logging
import os
import pickle
import random
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, ContextManager, List

import artm
import gensim.corpora as corpora
import numpy as np
import pandas as pd
from artm.batches_utils import BatchVectorizer, Batch
from billiard.exceptions import SoftTimeLimitExceeded
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm

from kube_fitness.metrics import MetricsScores, AVG_COHERENCE_SCORE, TimeMeasurements
from kube_fitness.utils import log_exec_timer

logger = logging.getLogger()
logging.basicConfig(level="INFO")


class SampleBatchVectorizer(BatchVectorizer):
    def __init__(self, sample_size=10, **kwargs):
        self.sample_size = sample_size
        super().__init__(**kwargs)

    def _parse_batches(self, data_weight, batches):
        if self._process_in_memory:
            self._model.master.import_batches(batches)
            self._batches_list = [batch.id for batch in batches]
            return

        data_paths, data_weights, target_folders = self._populate_data(data_weight, True)
        for (data_p, data_w, target_f) in zip(data_paths, data_weights, target_folders):
            if batches is None:
                batch_filenames = glob.glob(os.path.join(data_p, '*.batch'))
                batch_filenames = random.sample(batch_filenames, self.sample_size)
                self._batches_list += [Batch(filename) for filename in batch_filenames]

                if len(self._batches_list) < 1:
                    raise RuntimeError('No batches were found')

                self._weights += [data_w for i in range(len(batch_filenames))]
            else:
                self._batches_list += [Batch(os.path.join(data_p, batch)) for batch in batches]
                self._weights += [data_w for i in range(len(batches))]


# additional scores to calculate
# Topic Significance
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# Uniform Distribution Over Words (W-Uniform)
def ts_uniform(topic_word_dist):
    n_words = topic_word_dist.shape[0]
    w_uniform = np.ones(n_words) / n_words
    uniform_distances_kl = [kl_divergence(p, w_uniform) for p in topic_word_dist.T]
    return uniform_distances_kl


# Vacuous Semantic Distribution (W-Vacuous)
def ts_vacuous(doc_topic_dist, topic_word_dist, total_tokens):
    n_words = topic_word_dist.shape[0]
    #     n_tokens = np.sum([len(text) for text in texts])
    p_k = np.sum(doc_topic_dist, axis=1) / total_tokens
    w_vacauous = np.sum(topic_word_dist * np.tile(p_k, (n_words, 1)), axis=1)
    vacauous_distances_kl = [kl_divergence(p, w_vacauous) for p in topic_word_dist.T]
    return vacauous_distances_kl


# Background Distribution (D-BGround)
def ts_bground(doc_topic_dist):
    n_documents = doc_topic_dist.shape[1]
    d_bground = np.ones(n_documents) / n_documents
    d_bground_distances_kl = [kl_divergence(p.T, d_bground) for p in doc_topic_dist]
    return d_bground_distances_kl


# SwitchP
def switchp(phi, texts):
    words = phi.index.to_list()
    max_topic_word_dist = np.argmax(phi.to_numpy(), axis=1)
    max_topic_word_dist = dict(zip(words, max_topic_word_dist))
    switchp_scores = []
    for text in texts:
        mapped_text = [max_topic_word_dist[word] for word in text.split() if word in max_topic_word_dist]
        switches = (np.diff(mapped_text) != 0).sum()
        switchp_scores.append(switches / (len(mapped_text) - 1))

    return switchp_scores


def calculate_npmi(splitted_texts, id2word, topics_dict, num_tokens=50) -> Optional[List[float]]:
    try:
        specific_topics = [topic for topic in list(topics_dict.keys()) if topic.startswith('main')]

        topics_list = []
        for topic_i in specific_topics:
            topic_words = topics_dict[topic_i][:num_tokens]
            topics_list.append(topic_words)

        cm = CoherenceModel(topics=topics_list, dictionary=id2word, texts=splitted_texts, coherence='c_npmi')
        npmi_values_list = cm.get_coherence_per_topic()
    except Exception:
        logger.error(f"Failed to calculate npmi for num_tokens {num_tokens}", exc_info=True)
        npmi_values_list = None

    return npmi_values_list


class Dataset:
    _batches_path: str = 'batches'
    _wv_path: str = 'test_set_data_voc.txt'
    _cooc_dict_path: str = 'cooc_dictionary.txt'
    _dictionary_path: str = 'dictionary.txt'
    _vocab_path: str = 'vocab.txt'
    _cooc_file_df_path: str = 'cooc_df.txt'
    _cooc_file_tf_path: str = 'cooc_tf.txt'
    _ppmi_dict_df_path: str = 'ppmi_df.txt'
    _ppmi_dict_tf_path: str = 'ppmi_tf.txt'
    _mutual_info_dict_path: str = 'mutual_info_dict.pkl'
    _texts_path: str = 'dataset_processed.csv'
    _labels_path = "labels.pkl"

    def __init__(self, base_path: str, topic_count: int):
        self._base_path = base_path
        self._topic_count = topic_count

        self._labels = None
        self._dictionary: Optional[artm.Dictionary] = None
        self._batches: Optional[artm.BatchVectorizer] = None
        self._mutual_info_dict: Optional[Dict] = None

    @property
    def base_path(self) -> str:
        """
        :return: The path to the directory that contains this dataset files
        """
        return self._base_path

    @property
    def topic_count(self) -> int:
        return self._topic_count

    @property
    def labels(self):
        return self._labels

    @property
    def dictionary(self) -> artm.Dictionary:
        assert self._dictionary
        return self._dictionary

    @property
    def batches(self) -> artm.BatchVectorizer:
        assert self._batches
        return self._batches

    @property
    def mutual_info_dict(self) -> Dict:
        assert self._mutual_info_dict
        return self._mutual_info_dict

    @property
    def sample_batches(self) -> SampleBatchVectorizer:
        batches_dir_path = self._make_path(self._batches_path)
        return SampleBatchVectorizer(data_path=batches_dir_path, data_format='batches')

    @property
    def texts(self) -> List[str]:
        """
        :return: a list of PROCESSED texts of the corpus
        """
        texts_df_path = self._make_path(self._texts_path)
        df = pd.read_csv(texts_df_path)
        return df['processed_text'].tolist()

    @property
    def total_tokens(self) -> int:
        wv_path = self._make_path(self._wv_path)
        with open(wv_path) as f:
            total_tokens = sum((sum([int(elem.split(':')[1]) for elem in line.split()[1:]]) for line in f))
        return total_tokens

    def verify_dataset_files(self):
        """
        Verifies that mandatory files exist in the dataset folder defined by :py:func:'.base_path'
        """
        paths = [
            self._make_path(self._dictionary_path),
            self._make_path(self._batches_path),
            self._make_path(self._mutual_info_dict_path),
            self._make_path(self._texts_path)
        ]

        for p in paths:
            assert os.path.exists(p), f"Dataset file on path {p} should exist, but it doesn't"

    def load_dataset(self):
        """
        Partially loads this dataset files creating supplementary data_generator structures and entities like artm.BatchVectorizer
        """
        logger.info("Loading dataset entities")

        self.verify_dataset_files()

        start = time.time()
        labels_path = self._make_path(self._labels_path)
        dictionary_path = self._make_path(self._dictionary_path)
        batches_dir_path = self._make_path(self._batches_path)
        mutual_info_dict_path = self._make_path(self._mutual_info_dict_path)

        if os.path.exists(labels_path):
            logger.info(f"Reading labels {labels_path}")
            with open(labels_path, 'rb') as f:
                self._labels = pickle.load(f)

        logger.info(f"Reading dictionary from {dictionary_path}")
        self._dictionary = artm.Dictionary()
        self._dictionary.load_text(dictionary_path)
        # TODO: tune the params
        self._dictionary.filter(min_df=5, max_tf=1000)

        self._batches = artm.BatchVectorizer(data_path=batches_dir_path, data_format='batches')

        with open(mutual_info_dict_path, 'rb') as handle:
            self._mutual_info_dict = pickle.load(handle)

        logging.info(f"Dataset entities initialization took {time.time() - start: .2f}")

    def _make_path(self, path):
        """
        Creates a full path out of a relative path or a filename
        :param path: a relative path or a filename
        :return: a full path
        """
        return os.path.join(self._base_path, path)


class TopicModelFactory:
    num_processors: Optional[int] = None
    experiments_path: str = "/tmp/tm_experiments"
    cached_dataset_settings: Dict[str, Dataset] = dict()

    @classmethod
    def init_factory_settings(cls,
                              num_processors: Optional[int] = None,
                              dataset_settings: Dict[str, Dict[str, object]] = None):
        cls.num_processors = num_processors
        cls.cached_dataset_settings = {k: cls.init_dataset(v) for k, v in dataset_settings.items()} \
            if dataset_settings else dict()

    @classmethod
    def init_dataset(cls, settings) -> Dataset:
        assert 'base_path' in settings and 'topic_count' in settings
        dataset = Dataset(base_path=settings['base_path'], topic_count=settings['topic_count'])
        dataset.load_dataset()
        return dataset

    def __init__(self, dataset_name: str, fitness_name: str, params: list,
                 topic_count: Optional[int] = None, forced_update: bool = False,
                 train_option: str = 'offline'):
        self.dataset_name = dataset_name
        self.fitness_name = fitness_name
        self.params = params
        self.topic_count = topic_count
        self.forced_update = forced_update
        self.train_option = train_option
        self.tm = None

    def __enter__(self) -> 'TopicModel':
        if self.dataset_name not in self.cached_dataset_settings:
            raise Exception(f"No settings for dataset {self.dataset_name}")

        dataset = self.cached_dataset_settings[self.dataset_name]
        t_count = self.topic_count if self.topic_count else dataset.topic_count

        logging.debug(f"Using the following settings: \n{dataset.base_path}")

        uid = uuid.uuid4()

        if self.fitness_name == "default":
            logging.info(f"Using TM model: {TopicModel} according "
                         f"to fitness name: {self.fitness_name}, topics count: {t_count}")
            self.tm = TopicModel(uid, self.experiments_path, t_count, self.num_processors, dataset,
                                 self.params, train_option=self.train_option)
        else:
            raise Exception(
                f"Unknown fitness name: {self.fitness_name}. Only the following ones are known: {['default']}")

        self.tm.init_model()

        return self.tm

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tm.dispose()


def type_check(res):
    res = list(res)
    for i in [1, 4, 7, 10, 11]:
        res[i] = int(res[i])
    return res


@contextmanager
def fit_tm_of_individual(dataset: str,
                         params: list,
                         fitness_name: str = "default",
                         topic_count: Optional[int] = None,
                         force_dataset_settings_checkout: bool = False,
                         train_option: str = 'offline') \
        -> ContextManager[Tuple[TimeMeasurements, MetricsScores, 'TopicModel']]:
    params = type_check(params)

    start = time.time()

    with TopicModelFactory(dataset, fitness_name, params, topic_count, force_dataset_settings_checkout) as tm:
        try:
            with log_exec_timer("TM Training") as train_timer:
                tm.train()
            with log_exec_timer("Metrics calculation") as metrics_timer:
                fitness = tm.metrics_get_last_avg_vals(texts=tm.dataset.texts, total_tokens=tm.dataset.total_tokens)

            time_metrics = {
                "train": train_timer.duration,
                "metrics": metrics_timer.duration
            }
        except SoftTimeLimitExceeded as ex:
            raise ex
        except Exception:
            logger.exception(msg="Fitness calculation problem")
            fitness = {AVG_COHERENCE_SCORE: 0.0}
            time_metrics = {
                "train": -1,
                "metrics": -1
            }

        logging.info(f"Fitness estimation took {time.time() - start: .2f}")

        yield time_metrics, fitness, tm


def calculate_fitness_of_individual(dataset: str,
                                    params: list,
                                    fitness_name: str = "default",
                                    topic_count: Optional[int] = None,
                                    force_dataset_settings_checkout: bool = False,
                                    train_option: str = 'offline') -> MetricsScores:
    with fit_tm_of_individual(dataset, params, fitness_name, topic_count, force_dataset_settings_checkout,
                              train_option) as result:
        time_metrics, fitness, tm = result

    return fitness


class TopicModel:
    def __init__(self,
                 uid: uuid.UUID,
                 experiments_path: str,
                 topic_count: int,
                 num_processors: int,
                 dataset: Dataset,
                 params: list,
                 decor_test=False,
                 train_option: str = 'offline'):
        self.uid = uid
        self.experiments_path: str = experiments_path
        self.topic_count: int = topic_count
        self.num_processors: int = num_processors
        self.dataset = dataset
        self.train_option = train_option

        self.model = None
        self.S = self.topic_count
        self.specific = ['main{}'.format(i) for i in range(self.S)]

        self.save_path = os.path.join(self.experiments_path, 'best_model')
        self.decor_test = decor_test

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.__set_params(params)
        self.back = ['back{}'.format(i) for i in range(self.B)]

    def init_model(self):
        self.model = artm.ARTM(
            num_topics=self.S + self.B,
            class_ids=['@default_class'],
            dictionary=self.dataset.dictionary,
            show_progress_bars=False,
            cache_theta=True,
            topic_names=self.specific + self.back,
            num_processors=self.num_processors
        )

        self.__set_model_scores()

    def _early_stopping(self):
        coherences_main, coherences_back = self.__return_all_tokens_coherence(self.model, s=self.S, b=self.B)
        if len(coherences_main) < self.S or not any(coherences_main):
            return True
        return False

    def train(self, option='offline'):
        if self.model is None:
            print('Initialise the model first!')
            return

        self.model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorr',
                                                                    topic_names=self.specific, tau=self.decor))
        self.model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorr_2',
                                                                    topic_names=self.back, tau=self.decor_2))
        if option == 'offline':
            self.model.fit_offline(batch_vectorizer=self.dataset.batches, num_collection_passes=self.n1)
        elif option == 'online_v1':
            self.model.fit_offline(batch_vectorizer=self.dataset.sample_batches, num_collection_passes=self.n1)
        elif option == 'online_v2':
            self.model.num_document_passes = self.n1
            self.model.fit_online(batch_vectorizer=self.dataset.sample_batches,
                                  update_every=self.num_processors)

        if self.n1 > 0:
            if self._early_stopping():
                print('Early stopping is triggered')
                return

        #         if ((self.n2 != 0) and (self.B != 0)):
        if self.B != 0:
            self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SmoothPhi',
                                                                          topic_names=self.back, tau=self.spb))
            self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SmoothTheta',
                                                                          topic_names=self.back, tau=self.stb))
            if option == 'offline':
                self.model.fit_offline(batch_vectorizer=self.dataset.batches, num_collection_passes=self.n2)
            elif option == 'online_v1':
                self.model.fit_offline(batch_vectorizer=self.dataset.sample_batches, num_collection_passes=self.n2)
            elif option == 'online_v2':
                self.model.num_document_passes = self.n2
                self.model.fit_online(batch_vectorizer=self.dataset.sample_batches,
                                      update_every=self.num_processors)

        if self.n1 + self.n2 > 0:
            if self._early_stopping():
                print('Early stopping is triggered')
                return

        if self.n3 != 0:
            self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparsePhi',
                                                                          topic_names=self.specific, tau=self.sp1))
            self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta',
                                                                          topic_names=self.specific, tau=self.st1))
            if option == 'offline':
                self.model.fit_offline(batch_vectorizer=self.dataset.batches, num_collection_passes=self.n3)
            elif option == 'online':
                self.model.fit_offline(batch_vectorizer=self.dataset.sample_batches, num_collection_passes=self.n3)
            elif option == 'online_v2':
                self.model.num_document_passes = self.n3
                self.model.fit_online(batch_vectorizer=self.dataset.sample_batches,
                                      update_every=self.num_processors)

        if self.n1 + self.n2 + self.n3 > 0:
            if self._early_stopping():
                print('Early stopping is triggered')
                return

        if self.n4 != 0:
            self.model.regularizers['SparsePhi'].tau = self.sp2
            self.model.regularizers['SparseTheta'].tau = self.st2
            if option == 'offline':
                self.model.fit_offline(batch_vectorizer=self.dataset.batches, num_collection_passes=self.n4)
            elif option == 'online_v1':
                self.model.fit_offline(batch_vectorizer=self.dataset.sample_batches, num_collection_passes=self.n4)
            elif option == 'online_v2':
                self.model.num_document_passes = self.n4
                self.model.fit_online(batch_vectorizer=self.dataset.sample_batches,
                                      update_every=self.num_processors)

        if self.n1 + self.n2 + self.n3 > 0:
            if self._early_stopping():
                print('Early stopping is triggered')
                return

        print('Training is complete')

    def decor_train(self):
        if self.model is None:
            print('Initialise the model first')
            return

        self.model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorr',
                                                                    topic_names=self.specific, tau=self.decor))

    def dump_model(self, ii):
        self.model.dump_artm_model(os.path.join(self.save_path, 'model_{}'.format(ii)))

    def save_model(self, path):
        self.model.dump_artm_model(path)

    def print_topics(self):
        for i, (topic, top_tokens) in enumerate(self.get_topics().items()):
            print(topic)
            print(top_tokens)
            print()

    def get_topics(self):
        if 'TopTokensScore' not in self.model.score_tracker:
            logger.warning(f"Key 'TopTokensScore' is not presented in the model's (uid={self.uid}) score_tracker. "
                           f"Returning empty dict of topics.")
            return dict()
        res = self.model.score_tracker['TopTokensScore'].last_tokens
        topics = {topic: tokens[:50] for topic, tokens in res.items()}
        return topics

    def _get_avg_coherence_score(self, for_individ_fitness=False):
        coherences_main, coherences_back = self.__return_all_tokens_coherence(self.model, s=self.S, b=self.B)
        if for_individ_fitness:
            # print('COMPONENTS: ', np.mean(list(coherences_main.values())), np.min(list(coherences_main.values())))
            return np.mean(list(coherences_main.values())) + np.min(list(coherences_main.values()))
        return np.mean(list(coherences_main.values()))

    # added
    def _calculate_labels_coeff(self, topk=3):
        theta = self.model.get_theta()
        # theta.set_index('Unnamed: 0', inplace=True)
        documents_topics = OrderedDict()
        topics_documents = OrderedDict()
        for col in theta.columns:
            topics = theta.nlargest(topk, col).index.tolist()
            documents_topics[col] = topics
            for tp in topics:
                if tp not in topics_documents:
                    topics_documents[tp] = []
                topics_documents[tp].append(col)
        total_res = []
        for label in self.dataset.labels:
            docs = self.dataset.labels[label]
            all_lens = []
            for topic in topics_documents:
                all_lens.append(len(set(topics_documents[topic]).intersection(set(docs))))
            total_res.append(max(all_lens) / len(docs))
        coeff = np.mean(total_res)
        logging.info(f"Calculated coefficient: {coeff}")
        return coeff

    def dispose(self):
        """
        Disposes the model instance and removes the model from the object making it reusable again
        :return:
        """
        self.model.dispose()
        self.model = None

        log_files = [file for file in os.listdir(".") if file.startswith("bigartm.")]
        logging.info(f"Deleting bigartm logs: {log_files}")
        for file in log_files:
            os.remove(file)

    def __set_params(self, params_string):
        self.decor = params_string[0]
        self.n1 = params_string[1]

        if self.decor_test:
            return

        self.spb = params_string[2]
        self.stb = params_string[3]
        self.n2 = params_string[4]
        self.sp1 = params_string[5]
        self.st1 = params_string[6]
        self.n3 = params_string[7]
        self.sp2 = params_string[8]
        self.st2 = params_string[9]
        self.n4 = params_string[10]
        self.B = params_string[11]
        self.decor_2 = params_string[15]

    def __set_model_scores(self):

        self.model.scores.add(artm.PerplexityScore(name='PerplexityScore', dictionary=self.dataset.dictionary))

        self.model.scores.add(
            artm.SparsityPhiScore(name='SparsityPhiScore', class_id='@default_class', topic_names=self.specific))
        self.model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore', topic_names=self.specific))

        # Fraction of background words in the whole collection
        self.model.scores.add(
            artm.BackgroundTokensRatioScore(name='BackgroundTokensRatioScore', class_id='@default_class'))

        # Kernel characteristics
        self.model.scores.add(
            artm.TopicKernelScore(name='TopicKernelScore', class_id='@default_class', topic_names=self.specific,
                                  probability_mass_threshold=0.5, dictionary=self.dataset.dictionary))

        # Looking at top tokens
        self.model.scores.add(artm.TopTokensScore(name='TopTokensScore', class_id='@default_class', num_tokens=100))

    def __calculate_topic_coherence(self, tokens, top=50):
        tokens = tokens[:top]
        total_sum = 0
        for ix, token_1 in enumerate(tokens[:-1]):
            for ij, token_2 in enumerate(tokens[(ix + 1):]):
                try:
                    total_sum += self.dataset.mutual_info_dict['{}_{}'.format(token_1, token_2)]
                except KeyError:
                    total_sum += 0

        coherence = 2 / (top * (top - 1)) * total_sum
        return coherence

    def __return_all_tokens_coherence(self, model, s, b, top=50, return_backs=True):
        topics = list(model.score_tracker['TopTokensScore'].last_tokens.keys())

        res = model.score_tracker['TopTokensScore'].last_tokens

        topics_main = [topic for topic in topics if topic.startswith('main')]
        topics_back = [topic for topic in topics if topic.startswith('back')]

        all_topics_main = [i for i in range(s)]
        existing_topics_main = [int(i[4:]) for i in topics_main]
        inexisting_topics_main = [i for i in all_topics_main if i not in existing_topics_main]

        all_topics_back = [i for i in range(b)]
        existing_topics_back = [int(i[4:]) for i in topics_back]
        inexisting_topics_back = [i for i in all_topics_back if i not in existing_topics_back]

        coh_vals_main = {}
        # coherence for main topics
        for i, topic in tqdm(enumerate(topics_main)):
            coh_vals_main[topic] = self.__calculate_topic_coherence(res[topic][:50], top=top)
        for i, topic in tqdm(enumerate(inexisting_topics_main)):
            coh_vals_main['main{}'.format(i)] = 0

        coh_vals_back = {}
        # coherence for back topics
        for i, topic in tqdm(enumerate(topics_back)):
            coh_vals_back[topic] = self.__calculate_topic_coherence(res[topic][:50], top=top)
        for i, topic in tqdm(enumerate(inexisting_topics_back)):
            coh_vals_back['back{}'.format(i)] = 10  # penalty for not creating backs

        if return_backs:
            return coh_vals_main, coh_vals_back
        else:
            return coh_vals_main

    # TODO: fix
    def __return_all_coherence_types(self, model, S, only_specific=True, top=(10, 15, 20, 25, 30, 35, 40, 45, 50, 55)):
        topics = list(model.score_tracker['TopTokensScore'].last_tokens.keys())
        coh_vals = {}

        res = model.score_tracker['TopTokensScore'].last_tokens
        if only_specific:
            topics = [topic for topic in topics if topic.startswith('main')]

        all_topics = [i for i in range(S)]
        existing_topics = [int(i[4:]) for i in topics]
        inexisting_topics = [i for i in all_topics if i not in existing_topics]

        for num_tokens in top:
            coh_vals['coherence_{}'.format(num_tokens)] = {}
            for i, topic in tqdm(enumerate(topics)):
                coh_vals['coherence_{}'.format(num_tokens)][topic] = \
                    self.__calculate_topic_coherence(res[topic][:100], top=num_tokens)
            for i, topic in tqdm(enumerate(inexisting_topics)):
                coh_vals['coherence_{}'.format(num_tokens)][topic] = 0
        return coh_vals

    def metrics_get_avg_coherence_score(self, for_individ_fitness=False) -> MetricsScores:
        coherences_main, coherences_back = self.__return_all_tokens_coherence(self.model, s=self.S, b=self.B)
        # commented after Masha's consultation
        # coeff = self._calculate_labels_coeff()
        coeff = 1.0
        if for_individ_fitness:
            print('COMPONENTS: ', np.mean(list(coherences_main.values())), np.min(list(coherences_main.values())))
            avg_coherence_score = \
                np.mean(list(coherences_main.values())) + np.min(list(coherences_main.values())) * coeff
        else:
            avg_coherence_score = np.mean(list(coherences_main.values())) * coeff

        return {
            AVG_COHERENCE_SCORE: avg_coherence_score
        }

    # TODO: fix
    def metrics_get_last_avg_vals(self, texts, total_tokens,
                                  calculate_significance=False,
                                  calculate_npmi=False,
                                  calculate_switchp=False) -> MetricsScores:

        if calculate_significance:
            # turn off significance calculation
            phi_matrix = self.model.get_phi()
            theta_matrix = self.model.get_theta()

            phi_matrix = phi_matrix.to_numpy()
            theta_matrix = theta_matrix.to_numpy()
            topic_word_dist = phi_matrix[:, :self.S]  # getting subject scores
            doc_topic_dist = theta_matrix[:self.S, :]  #

            topic_significance_uni = np.mean(ts_uniform(topic_word_dist))
            topic_significance_vacuous = np.mean(ts_vacuous(doc_topic_dist, topic_word_dist, total_tokens))
            topic_significance_back = np.mean(ts_bground(doc_topic_dist))
            print(f'Topic Significance - Uniform Distribution Over Words: {topic_significance_uni}')
            print(f'Topic Significance - Vacuous Semantic Distribution: {topic_significance_vacuous}')
            print(f'Topic Significance - Background Distribution: {topic_significance_back}')
        else:
            topic_significance_uni = None
            topic_significance_vacuous = None
            topic_significance_back = None

        topic_names = list(self.model.score_tracker['TopTokensScore'].last_tokens.keys())
        topics_dict = self.model.score_tracker['TopTokensScore'].last_tokens

        specific_topics = [topic for topic in topic_names if topic.startswith('main')]

        all_topics_flag = False
        if len(specific_topics) == self.S:
            print('Wow! all topics')
            all_topics_flag = True

        logger.info("Building dictionary")

        if calculate_npmi:

            splitted_texts = [text.split() for text in texts]
            id2word = corpora.Dictionary(splitted_texts)

            logger.info("Calculating NPMIs")

            npmis = (
                (num_tokens, calculate_npmi(splitted_texts, id2word, topics_dict, num_tokens))
                for num_tokens in [15, 25, 50]
            )

            npmis = {
                k: v
                for num_tokens, npmi_list in npmis
                for k, v in [
                    (f"npmi_{num_tokens}", np.mean(npmi_list) if npmi_list else None),
                    (f"npmi_{num_tokens}_list", npmi_list)
                ]
            }

        else:
            npmis = {f"npmi_50": None,
                     f"npmi_15": None,
                     f"npmi_25": None,
                     f"npmi_50_list": None}

        if calculate_switchp:
            logger.info("Calculating switchp")

            if texts:
                switchp_scores = switchp(self.model.get_phi(), texts)
                switchp_scores = [score if score is not None else 0 for score in switchp_scores]
                print(f'SwitchP mean: {np.mean(switchp_scores)}')
            else:
                switchp_scores = None
        else:
            switchp_scores = [np.nan]

        background_ratio = self.model.score_tracker['BackgroundTokensRatioScore'].last_value
        control_perplexity = self.model.score_tracker['PerplexityScore'].last_value
        sparsity_phi = self.model.score_tracker['SparsityPhiScore'].last_value
        sparsity_theta = self.model.score_tracker['SparsityThetaScore'].last_value
        contrast = self.model.score_tracker['TopicKernelScore'].last_average_contrast
        purity = self.model.score_tracker['TopicKernelScore'].last_average_purity
        kernel_size = self.model.score_tracker['TopicKernelScore'].last_average_size

        avg_coherence_score = self.metrics_get_avg_coherence_score(for_individ_fitness=True)
        coherences = self.__return_all_coherence_types(self.model, self.S, only_specific=True)

        coherence_scores = dict()
        for i in range(10, 60, 5):
            coherence = list(coherences[f'coherence_{i}'].values())
            coherence_scores.update({f'coherence_{i}_list': coherence, f'coherence_{i}': np.mean(coherence)})

        scores_dict = {
            AVG_COHERENCE_SCORE: avg_coherence_score[AVG_COHERENCE_SCORE],
            'perplexityScore': control_perplexity,
            'backgroundTokensRatioScore': background_ratio,
            'contrast': contrast,
            'purity': purity,
            'kernelSize': kernel_size,
            'npmi_50_list': npmis['npmi_50_list'],  # npmi_values_50_list,
            'npmi_50': npmis['npmi_50'],
            'sparsity_phi': sparsity_phi,
            'sparsity_theta': sparsity_theta,
            'topic_significance_uni': topic_significance_uni,
            'topic_significance_vacuous': topic_significance_vacuous,
            'topic_significance_back': topic_significance_back,
            'switchP_list': switchp_scores,
            'switchP': np.nanmean(switchp_scores),
            'all_topics': all_topics_flag,
            **coherence_scores,
            **npmis
        }

        return scores_dict

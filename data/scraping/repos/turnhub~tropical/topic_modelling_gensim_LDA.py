from typing import Dict, List, Set, Any, Tuple  # noqa # pylint: disable=unused-import

import pandas as pd
import numpy as np

from gensim.models import Phrases
from gensim.models.phrases import Phraser

import gensim.corpora as corpora

from gensim.models.ldamodel import LdaModel as LDA
from gensim.models import CoherenceModel

from tropical.models.topic_modelling_base import TopicModellingBase

from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from nltk.corpus import stopwords as nltk_stopwords


class TopicModellingGensimLDA(TopicModellingBase):
    """
    Tropical Topic Modelling Gensim - Latent Dirichlet Allocation.

    Parameters:
    ----------
            max_topics (int): The maximum number of topics to try
            min_topics (int): The minimum number of topics to try
            step (int): step size for number of topics

    """

    def __init__(self, max_topics: int = None, min_topics: int = None, step: int = None) -> None:

        super().__init__()
        self.uuid = ""  # Duck typed unique version identifier.
        self.__basic_stopwords = set(nltk_stopwords.words('english')) & spacy_stopwords
        self.__basic_stop_phrases = self.__basic_stopwords

        if max_topics is None:
            self.max_topics = 10
        else:
            self.max_topics = max_topics

        if min_topics is None:
            self.min_topics = 2
        else:
            self.min_topics = min_topics

        if step is None:
            self.step = 2
        else:
            self.step = step

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # Pre-processing

    @staticmethod
    def __tokenize_and_lower(utterances):
        """ Whitespace tokenize and lowercase"""
        return [utterance.lower().split() for utterance in utterances]

    # ToDo : create the regex for this function
    @staticmethod
    def __remove_urls(utterances):
        """Explicitly remove URLS from content"""   # Usually Spam
        pass

    # ToDo : create the regex for this function
    @staticmethod
    def __lemmatize(utterances):
        """Can be used to allow only certain POS to be used (POS tagger for that language required)"""
        pass

    def __build_ngrams(self,
                       tokenised_utterances,
                       stopwords,
                       min_count=2,
                       threshold=10,
                       delimiter: bytes = b'_'):

        bigram = Phrases(
            tokenised_utterances,
            min_count=min_count,
            threshold=threshold,
            # max_vocab_size=40000000,
            delimiter=delimiter,
            common_terms=stopwords
        )

        ngram = Phrases(
            bigram[tokenised_utterances],
            min_count=min_count,
            threshold=50,  # ToDo: experiment with this threshold.
            # max_vocab_size=40000000,
            delimiter=delimiter,
            common_terms=stopwords
        )

        # Export the trained model = use less RAM, faster processing, but model updates no longer possible.
        bigram_phraser = Phraser(bigram)
        ngram_phraser = Phraser(ngram)

        # bigrammed_utterances = [bigram_phraser[utterance] for utterance in tokenised_utterances]
        ngrammed_utterances = [ngram_phraser[bigram_phraser[utterance]] for utterance in tokenised_utterances]

        return ngrammed_utterances

    def __build_lda_model(self, ngrammed_utterances, num_topics):
        """ gensim implementation of Latent Dirichlet Allocation
        """
        # Create Dictionary
        id2word = corpora.Dictionary(ngrammed_utterances)

        # Create Corpus
        texts = ngrammed_utterances

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # Build LDA model
        lda_model = LDA(corpus=corpus,
                        id2word=id2word,
                        num_topics=num_topics,
                        random_state=42,
                        update_every=8,
                        chunksize=2048,
                        passes=10,
                        alpha='auto',
                        per_word_topics=True)

        return lda_model

    # TODO : Might want to consider calculating 'best' topic inside this function
    #   PROS : Don't have to pass multiple models into other functions
    #   CONS : Stuck with using coherence to evaluate topic model

    def __compute_coherence_values(self, ngrammed_utterances):
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
        coherence_values = []
        model_list = []

        start = self.min_topics
        limit = self.max_topics
        step = self.step

        variations = limit//step
        if variations > 10:
            print(f"Running {variations} variations of the topic model, this might take a few minutes")

        for num_topics in range(start, limit+step, step):
            print(f"Running Model for {num_topics} topics")
            model = self.__build_lda_model(ngrammed_utterances, num_topics)

            dictionary = corpora.Dictionary(ngrammed_utterances)
            texts = ngrammed_utterances

            model_list.append(model)
            coherence_model = CoherenceModel(model=model,
                                             texts=texts,
                                             dictionary=dictionary,
                                             coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())

        print(coherence_values)
        return model_list, coherence_values

    def __extract_topics(self, model_list, coherence_values):
        """

        """
        best_model = model_list[np.argmax(coherence_values)]
        best_model_coherence = max(coherence_values)

        return best_model.show_topics(formatted=False, num_topics=self.max_topics, num_words=10), best_model_coherence

    def analyse_dataframe(self, big_dataframe_raw: pd.DataFrame,
                          delimiter: bytes = b'_') -> List[Dict[str, Any]]:
        """
        Analyse the messages in the incoming dataframe and extract topics.

        :param big_dataframe_raw: The input dataframe {'time_frame':, 'uuid':, 'content':}
        :param delimiter:
        :return: A dict [{'time_frame':, 'num_utterances':, 'topic_index': ['topic_terms':[]]}]
        """

        # Remove any rows with NaNs and missing values.
        big_dataframe_raw = big_dataframe_raw.dropna()

        if big_dataframe_raw.columns[0] == 'day':
            new_columns = list(big_dataframe_raw.columns)
            new_columns[0] = 'time_frame'
            big_dataframe_raw.columns = new_columns

        # big_dataframe_raw['time_frame'] = 'one_time_frame_to_rule_them_all'

        big_dataframe_raw.insert(len(big_dataframe_raw.columns), 'utterance_length', big_dataframe_raw['content'].str.len())

        utterance_length_threshold = np.percentile(big_dataframe_raw['utterance_length'], 98.0)
        print(f"utterance_length_threshold = {utterance_length_threshold}")
        big_dataframe = big_dataframe_raw[big_dataframe_raw['utterance_length'] <= utterance_length_threshold]

        response_frame_list: List[Dict[str, Any]] = list()

        for time_frame, data_frame in big_dataframe.groupby('time_frame'):
            print("Working on Frame", time_frame)
            dataframe_utterances = list(data_frame['content'])
            num_dataframe_utterances = len(dataframe_utterances)

            # uuids = list(data_frame['uuid'])

            tokenized_dataframe_utterances = self.__tokenize_and_lower(dataframe_utterances)
            ngrammed_dataframe_utterances = self.__build_ngrams(tokenized_dataframe_utterances,
                                                                threshold=10,  # ToDo: experiment with this threshold.
                                                                stopwords=self.__basic_stopwords,
                                                                delimiter=delimiter)

            model_list, coherence_values = self.__compute_coherence_values(ngrammed_dataframe_utterances)
            topics, coherence = self.__extract_topics(model_list, coherence_values)

            topic_terms_float32 = [topic[1] for topic in topics]
            topic_terms_float64 = []

            for topics in topic_terms_float32:
                topics_float64 = []
                for word, score in topics:
                    topics_float64.append((word, float(score)))
                topic_terms_float64.append(topics_float64)

            response_frame: Dict[str, Any] = dict()
            response_frame['time_frame'] = time_frame
            response_frame['num_utterances'] = num_dataframe_utterances
            response_frame['topics'] = topic_terms_float64
            response_frame['coherence'] = coherence

            response_frame_list.append(response_frame)

        return response_frame_list

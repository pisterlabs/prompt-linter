#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy script for analyzing LDA models.

Collection of tools for analysis of topic modeling results from either LDA or Top2Vec.
"""
from typing import Dict, Union, Optional
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

from rdsmproj import utils
from rdsmproj.tm_lda._ctfidf import CTFIDF


def get_topic_vectors(tokenized_documents:list[list[str]],
                      corpus:list[list[tuple[int, int]]],
                      model) -> list[list[float]]:
    """
    Gets the topic vectors for the different documents for use with LDA topic distribution. This
    returns vectors of probabilities of topics for each document from the model.

    Parameters
    ----------
    documents: list[list[str]]
        Tokenized list of documents.

    corpus: list[list[int, float]]
        Stream of document vectors made up of lists of tuples with (word_id, frequency).

    model: gensim.models.basemodel.BaseTopicModel
        Pre-trained topic model. Currently supports LdaModel,
        LdaMulticore, LdaMallet, and LdaVowpalWabbit.

    Returns
    -------
    topic_vectors: list[list[float]]
        Vectors of topic probabilities for each document.
    """
    topic_vectors = []
    for doc, _ in enumerate(tokenized_documents):
        top_topics = model.get_document_topics(corpus[doc],minimum_probability=0.0)
        topic_vector = [top_topics[topic][1] for topic in range(model.num_topics)]
        topic_vectors.append(topic_vector)
    return topic_vectors

def pad_docs_per_topic(docs_per_topic:Dict[int, list[str]],
                       num_topics:int) -> Dict[int, list[str]]:
    """
    Pads the docs_per_topic as some topics have no documents as their top topic. For use in
    wordclouds and topic distribution figures.

    Parameters
    ----------
    docs_per_topic: Dict[int, list[str]]
        Dictionary with each key being the topic number and the value being the list of documents
        that have that topic as their top topic based on probability.

    num_topics: int
        Number of topics in the model.

    Returns
    -------
    docs_per_topic
    """
    for topic in range(num_topics):
        # If topic is not one of the keys, then add it as an empty list.
        if topic not in docs_per_topic:
            docs_per_topic[topic] = []
    return docs_per_topic

def cluster_by_topic(documents:list[str],
                     topic_vectors:list[list[float]],
                     num_topics:int) -> Dict[int, list[str]]:
    """
    Clusters documents by assigning a top topic to that document based on the most probable topic.

    Parameters
    ----------
    documents: list[str]
        Filtered documents created from joining the tokens for a document together into one string.

    topic_vectors: list[list[float]]
        Vectors of topic probabilities for each document.

    num_topics: int
        Number of topics in the model.

    Returns
    -------
    docs_per_topic: Dict[int, list[str]]
        Dictionary with each key being the topic number and the value being the list of documents
        that have that topic as their top topic based on probability.
    """
    docs_per_topic = {}
    for doc, _ in enumerate(topic_vectors):
        # Finds the index of highest topic probability for the document.
        topic = topic_vectors[doc].index(max(topic_vectors[doc]))
        # Checks if topic is already a key in docs_per_topic and adds document to the value (list).
        if topic in docs_per_topic:
            docs_per_topic[topic].append(documents[doc])
        # If the topic is not already a key, then it creates one with [document] as the value.
        else:
            docs_per_topic[topic] = [documents[doc]]
    # Pads the dictionary with empty lists for topics with no documents.
    docs_per_topic = pad_docs_per_topic(docs_per_topic, num_topics)
    # Returns sorted dictionary with topics with the most documents first.
    return dict(sorted(docs_per_topic.items(), key = lambda x: len(x[1]), reverse=True))

def create_filtered_documents(documents:list[list[str]]) -> list[str]:
    """
    Creates filtered documents by concatenating tokens for a document into a single string with
    spaces in between each token.

    Parameters
    ----------
    documents: list[list[str]]
        Tokenized list of documents.

    Returns
    -------
        Documents created from joining the tokens for a document together into one string.
    """

    return [ ' '.join(doc) for doc in documents]


def create_wordcloud(word_dict:Dict[int, Dict[str, float]],
                     topic:Union[int, str],
                     coherence_values_per_topic:Union[list[float], Dict[str, float]],
                     document_size:int,
                     dict_type:str,
                     name:str,
                     path:Union[str, Path]):
    """
    Creates WordCloud from a dictionary of words and scores. Labels the coherence value and number
    of documents found in that topic.

    Parameters
    ----------
    word_dict: Dict[int, Dict[str, float]]
        Dictionary of topic labels for keys and values being list of tuples of words and their
        scores.

    topic: Union[int, str]
        Label for topic used as key for word_dict[key]

    coherence_values_per_topic: Union[list[float], Dict[str, float]]
        Coherence values per topic. Either a list with the topic number being the index of the
        score or a dictionary where the key is the label and the score is the value.

    document_size: int
        Number of documents for the topic where it is the most probable topic for those documents.

    dict_type: str
        Type of word_dict, used for title and saving file (e.g. 'Topic Word' or 'c-TFIDF')

    name: str
        Name of the collection of documents for use in title and saving file (e.g. 'CysticFibrosis')

    path: str, Path
        Path to where the figure will be saved to.
    """
    wordcloud = WordCloud(width=1600,
            height=400,
            background_color='white',
            max_words=30).generate_from_frequencies(word_dict)
    plt.figure(figsize=(16,4))
    plt.imshow(wordcloud)
    coherence = coherence_values_per_topic[topic]
    title = f' {name} {dict_type} Topic: {topic} Coherence: {coherence:.3f} Size: {document_size}'
    plt.title(title, fontsize='large', weight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(Path(path, f'{name}_{dict_type}_{topic}_wordcloud.png'), dpi=300)
    plt.clf()
    plt.close('all')

def create_distplot(docs_per_topics:Dict[int, list[str]], name:str, path:Union[str,Path]):
    """
    Creates a barplot of the distribution of different topics and the number of documents where
    that topic is the most probable topic.

    Parameters
    ----------
    docs_per_topic: Dict[int, list[str]]
        Dictionary with each key being the topic number and the value being the list of documents
        that have that topic as their top topic based on probability.

    name: str
        Name of the collection of documents for use in title and saving file (e.g. 'CysticFibrosis')

    path: str, Path
        Path to where the figure will be saved to.
    """
    plt.figure(figsize=(16,9))
    if isinstance(docs_per_topics, dict):
        x = [f'{key}' for key in docs_per_topics]
        y = [len(value) if isinstance(value, list) else 0 for value in docs_per_topics.values()]
    else:
        x = [i for i in range(len(docs_per_topics))]
        y = docs_per_topics
    sns.barplot(x=x, y=y, color =[0,114/255,178/255])
    #plt.xticks(fontsize='x-small')
    plt.xticks(rotation='vertical')
    plt.xlabel('Topic Number')
    plt.ylabel('Number of Posts')
    plt.title(f'Topic Distribution for {name}', weight='bold', fontsize='large')
    plt.tight_layout()
    plt.savefig(Path(path, f'{name}_postdistplot.png'), dpi=300)
    plt.clf()
    plt.close('all')

def create_coherence_distplot(coherence_values_per_topic:Union[list[float],
                              Dict[str, float]],
                              name:str,
                              path:Union[str,Path]):
    """
    Create coherence distribution plot of the topics and their coherence values.

    Parameters
    ----------
    coherence_values_per_topic: Union[list[float], Dict[str, float]]
        Coherence values per topic. Either a list with the topic number being the index of the
        score or a dictionary where the key is the label and the score is the value.

    name: str
        Name of the collection of documents for use in title and saving file (e.g. 'CysticFibrosis')

    path: str, Path
        Path to where the figure will be saved to.
    """
    plt.figure(figsize=(4.5,8))
    sns.displot(coherence_values_per_topic, binwidth = 0.05)
    plt.axvline(x=np.mean(coherence_values_per_topic), label='Average Coherence', linestyle='--')
    plt.title(f'{name}\nTopic Coherence Distribution')
    plt.xlabel('Topic Coherence')
    plt.xlim([0.0, 1.05])
    plt.ylabel('Topic Count')
    plt.tight_layout()
    plt.savefig(Path(path,f'{name}_coherencedistplot.png'), dpi=300)
    plt.clf()
    plt.close('all')

def create_tfidf_topic_word_list(word_score_dict:Dict[int, Dict[str, float]],
                                 topic_word_score_dict) -> list[list[str]]:
    """
    Creates a word list of the top n words for each topic from a dictionary. For use in creating
    a coherence model for calculating coherence values.

    Parameters
    ----------
    word_score_dict: Dict[int, Dict[str, float]]
            Dictionary where keys are labels from topic_list and values are dictionaries of
            keys = word:str and values = score:float) for each label.

    Returns
    -------
    topic_word_list: list[list[str]]
        List of top n words for each topic from the top_n_words dictionary.
    """
    topic_word_list = []
    for topic in word_score_dict.keys():
        topic_list = list(word_score_dict[topic].keys())
        if len(topic_list) < 2:
            topic_list = list(topic_word_score_dict[topic].keys())
        topic_word_list.append(topic_list)
    return topic_word_list

def create_word_dict(topic_words:list[list[str]],
                     word_scores:list[list[float]],
                     topic_nums:list[int]) -> Dict[int, Dict[str, float]]:
    """
    Creates a dictionary of words and scores for each topic. For use with Top2Vec models to create
    word score dictionaries for wordcloud creation.

    Parameters
    ----------
    topic_words: list[list[str]]
        List of topics and their list of top words that make up that topic based on their scores.

    word_scores: list[list[float]]
        List of topics and the list of word scores that correspond to each word top word for that
        topic.

    topic_nums: list[int]
        List of topic numbers.

    Returns
    -------
    word_score_dict: Dict[int, Dict[str, float]]
            Dictionary where keys are labels from topic_list and values are dictionaries of
            keys = word:str and values = score:float) for each label.
    """
    word_score_dict = {}
    for topic in topic_nums:
        words = topic_words[topic]
        scores = [float(score) for score in word_scores[topic]]
        word_score_dict[int(topic)] = dict(zip(words, scores))
    return word_score_dict

def create_word_score_dict(probs:list[tuple[int, list[tuple[str, float]]]]
                           ) -> Dict[int, Dict[str,float]]:
    """
    Parameters
    ----------
    probs: list[tuple[int, list[tuple[str, float]]]]
        Probabilities for the topn words in each derived topic from a gensim model.

    Returns
    -------
    Dictionary of topics with the values being dictionaries of the words and their probabilities.
    """
    return {prob[0]:{word:float(probability) for word, probability in prob[1]} for prob in probs}

def create_topic_sizes_dict(topic_sizes:list[int]) -> Dict[str, int]:
    """
    Creates topic size distribution dictionary from the topic size list for Top2Vec models.

    Parameters
    ----------
    topic_sizes: list[int]

    Returns
    -------
    Dictionary of topics and the number of documents that have that topic as their primary document
    based on the highest topic score for that document.
    """
    return {topic: int(size) for topic, size in enumerate(topic_sizes)}

def find_distribution(model:gensim.models.basemodel.BaseTopicModel,
                      tokenized_docs:list[list[str]],
                      corpus:list[tuple[int, int]]) -> Dict[int, list[str]]:
    """
    Finds the distribution of posts for each topic.

    Parameters
    ----------
    model: gensim.models.basemodel.BaseTopicModel (Optional, default None)
        Pre-trained topic model provided if topics not provided. Currently supports LdaModel,
        LdaMulticore, LdaMallet, and LdaVowpalWabbit.

    tokenized_docs: list[list[str]]
        Tokenized list of documents.

    corpus: list[tuple[int, int]]
        Document vectors made up of list of tuples with (word_id, word_frequency)

    Returns
    -------
    docs_per_topic: Dict[int, list[str]]
        Dictionary with each key being the topic number and the value being the list of documents
        that have that topic as their top topic based on probability.
    """
    # This is to weed out models in which the topics have no documents associated with them.

    # Retrieves topic vectors from the model. Used to find the most probable topic for each
    # document.
    topic_vectors = get_topic_vectors(tokenized_documents = tokenized_docs,
                                    corpus=corpus,
                                    model=model)
    # Concatenates tokenized documents into single strings.
    f_documents = create_filtered_documents(tokenized_docs)
    # Clusters the documents by the most probable topic for each document.
    num_topics = model.num_topics
    docs_per_topic = cluster_by_topic(f_documents, topic_vectors, num_topics)
    docs_per_topic = {topic: len(docs) for topic, docs in docs_per_topic.items()}
    return docs_per_topic

def create_coherence_model(model:Optional[gensim.models.basemodel.BaseTopicModel] = None,
                           topics:Optional[list[list[str]]] = None,
                           texts:Optional[list[list[str]]] = None,
                           id2word:Optional[Dictionary] = None,
                           corpus:Optional[list[tuple[int, int]]] = None,
                           coherence:str = 'c_v',
                           topn:int = 10,
                           processes:int = 1):
    """
    Creates a gensim.models.coherencemodel.CoherenceModel object from either a model or list of
    tokenized topics. Used to calculate coherence of a topic.

    Parameters
    ----------
    model: gensim.models.basemodel.BaseTopicModel (Optional, default None)
        Pre-trained topic model provided if topics not provided. Currently supports LdaModel,
        LdaMulticore, LdaMallet, and LdaVowpalWabbit.

    topics: list[list[str]] (Optional, default None)
        List of tokenized topics. id2word must be provided.

    texts: list[list[str]] (Optional, default None)
        Tokenized texts for use with sliding window based probability estimator ('c_something').

    id2word: gensim.corpora.dictionary.Dictionary (Optional, default None)
        If model present, not needed. If both provided, passed id2word will be used.

    corpus: list[tuple[int, int]] (Optional, default None)
        Document vectors made up of list of tuples with (word_id, word_frequency)

    coherence: str (default 'c_v')
        Currently through gensim supports following coherence measures: 'u_mass', 'c_v', 'c_uci',
        and 'c_npmi. Coherence measure 'c_uci = 'c_pmi'.

    topn: int (default 10)
        Integer corresponding to number of top words to be extracted from each topic.

    processes: int (default 1)
        Number of processes to use, any value less than 1 will be num_cpus - 1.

    Returns
    -------
    coherence_model: gensim.models.coherencemodel.CoherenceModel
        CoherenceModel object used for building and maintaining a model for topic coherence.
    """
    coherence_model = CoherenceModel(model = model,
                                    topics = topics,
                                    texts = texts,
                                    dictionary= id2word,
                                    corpus = corpus,
                                    coherence = coherence,
                                    topn = topn,
                                    processes = processes)
    return coherence_model

class AnalyzeTopics:
    """
    Class to call analysis tools to create files and visualizations to analyze the results of topic
    modeling done.

    Parameters
    ----------
    model:
        Topic model. Currently either supports either model from gensim (LDA) or Top2Vec.

    name: str
        Name of the collection of documents for use in title and saving file (e.g. 'CysticFibrosis')

    tokenized_docs: list[list[str]]
        Tokenized list of documents.

    id2word: Dict[(int, str)]
        Mapping of word ids to words.

    corpus: list[tuple[int, int]]
        Document vectors made up of list of tuples with (word_id, word_frequency)

    model_type: str ('LDA', 'Top2Vec')
        Model type for model passed to class. Currently only supports gensim or Top2Vec models.

    coherence: str (default 'c_v')
        Currently through gensim supports following coherence measures: 'u_mass', 'c_v', 'c_uci',
        and 'c_npmi. Coherence measure 'c_uci = 'c_pmi'.

    path: Path, str (Optional, default None)
        Path to store the analysis results files to.
    """
    def __init__(self,
                 model,
                 subreddit_name:str,
                 model_name:str,
                 tokenized_docs:list[list[str]],
                 id2word:Dict[(int, str)],
                 corpus:list[tuple[int, int]],
                 model_type:str,
                 coherence:str='c_v',
                 path:Optional[Union[Path,str]]=None):

        self.model_name = model_name
        self.subreddit_name = subreddit_name
        self.tokenized_docs = tokenized_docs
        self.id2word = id2word
        self.corpus = corpus
        self.model_type = model_type

        # Sets the path for analysis files and plots to be saved to.
        if path is None:
            results_path = utils.get_data_path('results')
            self.path = Path(results_path, self.subreddit_name, self.model_name)
        else:
            self.path = path
        utils.check_folder(self.path)

        # Checks for model type. Currently only supports LDA and Top2Vec.
        if self.model_type == 'LDA':
            # Sets the model for use in analysis.
            self.model = model
            # Finds the number of derived topics from the model.
            num_topics = self.model.num_topics
            # Creates a coherence model used to calculate coherence values.
            coherence_model = create_coherence_model(model=self.model,
                                                texts=self.tokenized_docs,
                                                id2word=id2word,
                                                coherence=coherence)
            # Prints the number of topics and the mean coherence of derived topics.
            print(f'Num Topics: {num_topics} Coherence: {coherence_model.get_coherence()}')
            # Finds the coherence values for each derived topic.
            coherence_values_per_topic = coherence_model.get_coherence_per_topic()
            # Saves the coherence values for each derived topic.
            utils.dump_json(coherence_values_per_topic,
                      self.path,
                      f'{self.model_name}_coherence_values_per_topic_LDA')
            # Creates the coherence distribution plot of coherence values for each topic with the
            # dashed line showing the mean coherence value.
            create_coherence_distplot(coherence_values_per_topic,
                                      f'{self.model_name} LDA',
                                      self.path)
            # Retrieves the topn words for each topic.
            topics = coherence_model.top_topics_as_word_lists(model, id2word, topn=50)

            # Generates data for use in the WordCloud figures.

            # Retrieves probabilities for the topic words.
            probs = model.show_topics(num_topics=num_topics, num_words = 50, formatted = False)
            # Creates a word score dictionary of topic words and their probability.
            word_score_dict = create_word_score_dict(probs)
            # Saves the word score dictionary.
            utils.dump_json(word_score_dict,
                      self.path,
                      f'{self.model_name}_LDA_word_score_dict')

            # Retrieves topic vectors from the model. Used to find the most probable topic for each
            # document.
            topic_vectors = get_topic_vectors(tokenized_documents = self.tokenized_docs,
                                            corpus=self.corpus,
                                            model=self.model)
            # Concatenates tokenized documents into single strings.
            f_documents = create_filtered_documents(tokenized_docs)
            # Clusters the documents by the most probable topic for each document.
            docs_per_topic = cluster_by_topic(f_documents, topic_vectors, num_topics)
            # Saves the clustered documents.
            utils.dump_json(docs_per_topic,
                      self.path,
                      f'{self.model_name}_LDA_docs_per_topic')
            # Creates the document distribution from the clustered documents.
            create_distplot(docs_per_topic, self.model_name, self.path)

            # Creates the wordcloud figures from the word score dictionary. Includes data in the
            # figures from the number of documents in the topic as well as the coherence value for
            # that topic. Only creates figures if the topic has more than 10 documents.
            for topic in docs_per_topic:
                if len(docs_per_topic[topic]) >= 10:
                    create_wordcloud(word_score_dict[topic],
                                    topic,
                                    coherence_values_per_topic,
                                    len(docs_per_topic[topic]),
                                    'Topic Word',
                                    self.model_name,
                                    self.path)

            # c-TFIDF calculations for class (topic) based TFIDF for each topic using algorithm
            # by Maarten Grootendorst as part of BERTopic: https://github.com/MaartenGr/BERTopic.

            # Concatenates all documents into a single string based on their most probable derived
            # topic.
            clustered_docs = [ ' '.join(values) for values in docs_per_topic.values()]
            # Creates a word score dict from the c-TFIDF values for the top 50 words.
            tfidf_word_score_dict = CTFIDF(clustered_docs=clustered_docs,
                                           topic_list=list(docs_per_topic.keys()),
                                           n=50)._extract_words_per_topic(ngram_range=(1,1))
            # Saves the word score dict.
            utils.dump_json(tfidf_word_score_dict,
                      self.path,
                      f'{self.model_name}_LDA_tfidf_word_score_dict')
            # Retrieves the topic words derived from c-TFIDF for each topic.
            topic_word_list = create_tfidf_topic_word_list(tfidf_word_score_dict, word_score_dict)

            # Creates a coherence model using the c-TFIDF topic words for each topic.
            tfidf_c_model = create_coherence_model(topics=topic_word_list,
                                                   texts=tokenized_docs,
                                                   corpus=corpus,
                                                   id2word=id2word,
                                                   coherence=coherence)
            # Prints the number of topics and the mean coherence of derived topics for the model
            # using the c-TFIDF topic words.
            tfidf_coherence = tfidf_c_model.get_coherence()
            print(f'Num Topics: {num_topics} Coherence: {tfidf_coherence}')
            # Retrieves the coherence values for each derived topic for the model using the c-TFIDF
            # topic words.
            tfidf_coherence_values_per_topic = tfidf_c_model.get_coherence_per_topic()
            # Saves the coherence values for each topic.
            utils.dump_json(tfidf_coherence_values_per_topic,
                      self.path,
                      f'{self.model_name}_LDA_tfidf_coherence_values_per_topic')

            # Creates the coherence distribution plot of coherence values for each topic with the
            # dashed line showing the mean coherence value.
            create_coherence_distplot(tfidf_coherence_values_per_topic,
                                      f'{self.model_name} c-TFIDF',
                                      self.path)

            # Creates the wordcloud figures from the word score dictionary. Includes data in the
            # figures from the number of documents in the topic as well as the coherence value for
            # that topic. Only creates figures if the topic has more than 10 documents.
            topic_sizes = {}
            for topic in docs_per_topic:
                topic_sizes[topic] = len(docs_per_topic[topic])
                if topic_sizes[topic] >= 10:
                    create_wordcloud(tfidf_word_score_dict[topic],
                                    topic,
                                    tfidf_coherence_values_per_topic,
                                    topic_sizes[topic],
                                    'c-TFIDF',
                                    self.model_name,
                                    self.path)

            # Saves the number of documents for each topic.
            utils.dump_json(topic_sizes,
                      self.path,
                      f'{self.model_name}_topic_sizes_LDA')

        elif self.model_type == 'Top2Vec':
            # Sets the model for use in analysis.
            self.model = model
            # Retrieves topic sizes and numbers from the Top2Vec model.
            topic_sizes, topic_nums = self.model.get_topic_sizes()
            # Creates a dictionary of topic sizes.
            topic_sizes_dict = create_topic_sizes_dict(topic_sizes)
            # Saves the topic size dictionary.
            utils.dump_json(topic_sizes_dict,
                      self.path,
                      f'{self.model_name}_topic_sizes_Top2Vec')
            # Creates a distribtion plot of number of documents for each topic.
            create_distplot(topic_sizes, f'{self.model_name} Top2Vec', self.path)

            # Retrieves topic words and their scores from the model.
            topic_words, word_scores, _ = model.get_topics()
            # Creates a list of topic words for each topic for use in coherence model creation.
            topics = [list(words) for words in topic_words]
            # Creates a coherence model using the topic word list.
            coherence_model = create_coherence_model(topics=topics,
                                                    texts=tokenized_docs,
                                                    id2word=id2word,
                                                    coherence=coherence)

            # Prints the number of topics and the mean coherence of derived topics for the model.
            num_topics = self.model.get_num_topics()
            coherence = coherence_model.get_coherence()
            print(f'Model: {self.model_name} Num Topics: {num_topics} Coherence: {coherence}')
            # Retrieves the coherence values for each topic.
            coherence_values_per_topic = coherence_model.get_coherence_per_topic()
            # Saves the coherence values for each topic.
            utils.dump_json(coherence_values_per_topic,
                      self.path,
                      f'{self.model_name}_coherence_values_per_topic_Top2Vec')

            # Creates a dictionary of word scores. For use in creating wordclouds.
            word_score_dict = create_word_dict(topic_words, word_scores, topic_nums)
            # Saves the word score dictionary.
            utils.dump_json(word_score_dict,
                      self.path,
                      f'{self.model_name}_word_score_dict_Top2Vec')

            # Creates the wordcloud figures from the word score dictionary. Includes data in the
            # figures from the number of documents in the topic as well as the coherence value for
            # that topic. Only creates figures if the topic has more than 10 documents.
            for topic in topic_nums:
                if topic_sizes[topic] >= 10:
                    create_wordcloud(word_score_dict[topic],
                                    topic,
                                    coherence_values_per_topic,
                                    topic_sizes[topic],
                                    'Topic Word',
                                    f'{self.model_name}_Top2Vec',
                                    self.path)
            # Creates the coherence distribution plot of coherence values for each topic with the
            # dashed line showing the mean coherence value.
            create_coherence_distplot(coherence_values_per_topic,
                                      f'{self.model_name} Top2Vec',
                                      self.path)

        else:
            print(f'Model type: {self.model_type} is not valid. Use LDA or Top2Vec')

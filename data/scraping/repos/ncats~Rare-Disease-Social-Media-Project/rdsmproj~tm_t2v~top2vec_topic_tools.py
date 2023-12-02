#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of tools for analysis of topic modeling results from Top2Vec.
"""
from typing import Dict, Union, Optional
from pathlib import Path
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

from rdsmproj import utils


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

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 60)

def create_wordcloud_subplots(data:Dict[int, Dict[str, float]],
                              suptitle:str,
                              path:Union[str,Path],
                              topics:Optional[list[list[str]]] = None,
                              max_words:Optional[int]=50,
                              context:Optional[str]='paper'):
    """
    Creates wordcloud subplots for a top2vec model.

    Parameters
    ----------
    data: Dict[int, Dict[str, float]]
        Word score dict. Dictionary where keys are labels from topic_list and values are
        dictionaries of keys = word:str and values = score:float) for each label.
    suptitle: str
        Name of the collection of documents for use in title and saving file
        (e.g. 'CysticFibrosis')
    path: str, Path
        Path to where the figure will be saved to.
    topics: list[list[str]] (Optional, default None)
        List of tokenized topics. id2word must be provided.
    max_words: int (Optional, default 50)
        Maximum number of words for each wordcloud.
    context: str (Optional, default paper)
        Name of context to pass to seaborn for plot style.
    """
    cm = 1/2.54
    sns.set_context(context)
    sns.set_style(style='white')

    if topics is None:
        num_topics = len(data)
        topics = list(data.keys())
    else:
        num_topics = len(topics)

    if num_topics < 5:
        num_cols = num_topics
        fig_width = (16/5)*cm*num_cols
        num_rows = 1
    else:
        num_cols = 5
        fig_width = 16*cm
        num_rows = int(np.ceil(num_topics/5))

    widths = [400]*num_cols
    heights = [500]*num_rows

    fig_height = fig_width * sum(heights)/sum(widths)
    fig, axs = plt.subplots(num_rows,num_cols, figsize=(fig_width, fig_height),
                            gridspec_kw = {'height_ratios': heights, 'wspace':0, 'hspace':0},
                            constrained_layout=True)
    fig.suptitle(suptitle, weight='bold')
    for n, ax in enumerate(axs.flat):
        if n < len(topics):
            try:
                wordcloud = WordCloud(background_color='white', width=400, height=400,
                                    max_words=max_words
                                    ).generate_from_frequencies(data[topics[n]])
                ax.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
                        interpolation="bilinear")
                ax.text(0, -5, f'{topics[n]}',weight='bold')
                ax.axis('off')
            except OSError:
                ax.axis('off')
                pass
        else:
            ax.axis('off')

    plt.savefig(Path(path, f'{suptitle}_wordcloud.png'), dpi=300)
    plt.clf()
    plt.close('all')

class AnalyzeTopics:
    """
    Class to call analysis tools to create files and visualizations to analyze the results of topic
    modeling done.

    Parameters
    ----------
    model:
        Topic model. Currently either supports model from Top2Vec.

    subreddit_name: str
        Name of the collection of documents for use in title and saving file
        (e.g. 'r/CysticFibrosis')

    model_name: str
        Name of the embedding model.

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

        # Creates a dictionary of word scores. For use in creating wordclouds.
        word_score_dict = create_word_dict(topic_words, word_scores, topic_nums)
        # Saves the word score dictionary.
        utils.dump_json(word_score_dict,
                    self.path,
                    f'{self.model_name}_word_score_dict_Top2Vec')

        # Prints the number of topics and the mean coherence of derived topics for the model.
        num_topics = self.model.get_num_topics()

        coherence_value = {}
        coherence_values_per_topic = []
        for coherence_measure in ['c_v', 'c_npmi', 'u_mass', 'c_uci']:
            try:
                coherence_model = create_coherence_model(topics=topics,
                                                        texts=tokenized_docs,
                                                        id2word=id2word,
                                                        coherence=coherence_measure)
                coherence_value[coherence_measure] = coherence_model.get_coherence()
                #print(f'{coherence_measure}: {coherence_value[coherence_measure]}')
                if coherence_measure == coherence:
                    coherence_values_per_topic = coherence_model.get_coherence_per_topic()
                    # Saves the coherence values for each topic.
                    utils.dump_json(coherence_values_per_topic,
                        self.path,
                        f'{self.model_name}_coherence_values_per_topic_Top2Vec')

            except ValueError:
                coherence_value[coherence_measure] = np.nan
                #print(f'{coherence_measure}: {coherence_value[coherence_measure]}')

        if coherence_values_per_topic:
            # Saves the coherence value dictionary.
            utils.dump_json(coherence_value, self.path, f'{self.model_name}_coherence_values')

            # Creates the coherence distribution plot of coherence values for each topic with the
            # dashed line showing the mean coherence value.
            if num_topics > 1:
                create_coherence_distplot(coherence_values_per_topic,
                                        f'{self.model_name} Top2Vec',
                                        self.path)

            print(f'>>> Model: {self.model_name}')
            print(f'>>> Num Topics: {num_topics}')
            print(f'>>> Coherence ({coherence}): {coherence_value[coherence]}')

            # Creates wordcloud figure.
            if num_topics > 1:
                create_wordcloud_subplots(word_score_dict,
                                        suptitle = self.subreddit_name,
                                        path=self.path)
        else:
            print(f'No coherence model was created for {self.model_name}')

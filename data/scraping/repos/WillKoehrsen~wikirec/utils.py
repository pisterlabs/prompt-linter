"""
utils
-----

Utility functions for data loading and cleaning

Contents:
    _check_str_similarity,
    _check_str_args,
    graph_lda_topic_evals,
    english_names_list
"""

import os
from difflib import SequenceMatcher

import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel


def _check_str_similarity(str_1, str_2):
    """Checks the similarity of two strings"""
    return SequenceMatcher(None, str_1, str_2).ratio()


def _check_str_args(arguments, valid_args):
    """
    Checks whether a str argument is valid, and makes suggestions if not
    """
    if type(arguments) == str:
        if arguments in valid_args:
            return arguments

        else:
            suggestions = []
            for v in valid_args:
                similarity_score = round(
                    _check_str_similarity(str_1=arguments, str_2=v), 2
                )
                arg_and_score = (v, similarity_score)
                suggestions.append(arg_and_score)

            ordered_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)

            print(f"'{arguments}' is not a valid argument for the given function.")
            print(f"The closest valid options to '{arguments}' are:")
            for item in ordered_suggestions[:5]:
                print(item)

            return

    elif type(arguments) == list:
        # Check arguments, and remove them if they're invalid
        for a in arguments:
            _check_str_args(arguments=a, valid_args=valid_args)

        return arguments


def graph_lda_topic_evals(
    corpus=None,
    num_topic_words=10,
    topic_nums_to_compare=None,
    metrics=True,
    verbose=True,
    **kwargs,
):
    """
    Graphs metrics for the given models over the given number of topics

    Parameters
    ----------
        corpus : list of lists (default=None)
            The text corpus over which analysis should be done

        num_topic_words : int (default=10)
            The number of keywords that should be extracted

        topic_nums_to_compare : list (default=None)
            The number of topics to compare metrics over

            Note: None selects all numbers from 1 to num_topic_words

        metrics : str or bool (default=True: all metrics)
            The metrics to include

            Options:
                stability: model stability based on Jaccard similarity

                coherence: how much the words associated with model topics co-occur

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query

        **kwargs : keyword arguments
            Arguments correspoding to gensim.models.ldamulticore.LdaMulticore

    Returns
    -------
        ax : matplotlib axis
            A graph of the given metrics for each of the given models based on each topic number
    """
    assert (
        metrics == "stability" or metrics == "coherence" or metrics == True
    ), "An invalid value has been passed to the 'metrics' argument - please choose from 'stability', 'coherence', or True for both."

    if metrics == True:
        metrics = ["stability", "coherence"]

    if type(corpus[0]) != list:
        corpus = [c.split() for c in corpus]

    def jaccard_similarity(topic_1, topic_2):
        """
        Derives the Jaccard similarity of two topics

        Notes
        -----
            Jaccard similarity:
                - A statistic used for comparing the similarity and diversity of sample sets
                - J(A,B) = (A ∩ B)/(A ∪ B)
                - Goal is low Jaccard scores for coverage of the diverse elements
        """
        # Fix for cases where there are not enough texts for clustering models
        if topic_1 == [] and topic_2 != []:
            topic_1 = topic_2
        if topic_1 != [] and topic_2 == []:
            topic_2 = topic_1
        if topic_1 == [] and topic_2 == []:
            topic_1, topic_2 = ["_None"], ["_None"]
        intersection = set(topic_1).intersection(set(topic_2))
        num_intersect = float(len(intersection))

        union = set(topic_1).union(set(topic_2))
        num_union = float(len(union))

        return num_intersect / num_union

    plt.figure()  # begin figure

    dirichlet_dict = corpora.Dictionary(corpus)
    bow_corpus = [dirichlet_dict.doc2bow(text) for text in corpus]

    # Add an extra topic so that metrics can be calculated all inputs
    if topic_nums_to_compare == None:
        topic_nums_to_compare = list(range(num_topic_words + 1)[1:])
    else:
        topic_nums_to_compare.append(topic_nums_to_compare[-1] + 1)

    LDA_models = {}
    LDA_topics = {}
    disable = not verbose
    for i in tqdm(
        iterable=topic_nums_to_compare, desc="LDA models ran", disable=disable
    ):
        LDA_models[i] = LdaMulticore(
            corpus=bow_corpus, id2word=dirichlet_dict, num_topics=i, **kwargs
        )

        shown_topics = LDA_models[i].show_topics(
            num_topics=i, num_words=num_topic_words, formatted=False
        )
        LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]

    LDA_stability = {}
    for i in range(0, len(topic_nums_to_compare) - 1):
        jaccard_sims = []
        for t1, topic1 in enumerate(  # pylint: disable=unused-variable
            LDA_topics[topic_nums_to_compare[i]]
        ):
            sims = []
            for t2, topic2 in enumerate(  # pylint: disable=unused-variable
                LDA_topics[topic_nums_to_compare[i + 1]]
            ):
                sims.append(jaccard_similarity(topic1, topic2))

            jaccard_sims.append(sims)

        LDA_stability[topic_nums_to_compare[i]] = jaccard_sims

    mean_stabilities = [
        np.array(LDA_stability[i]).mean() for i in topic_nums_to_compare[:-1]
    ]

    coherences = [
        CoherenceModel(
            model=LDA_models[i],
            texts=corpus,
            dictionary=dirichlet_dict,
            coherence="c_v",
        ).get_coherence()
        for i in topic_nums_to_compare[:-1]
    ]

    coh_sta_diffs = [
        coherences[i] - mean_stabilities[i]
        for i in range(len(topic_nums_to_compare))[:-1]
    ]  # limit topic numbers to the number of keywords
    coh_sta_max = max(coh_sta_diffs)
    coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
    ideal_topic_num_index = coh_sta_max_idxs[
        0
    ]  # choose less topics in case there's more than one max
    ideal_topic_num = topic_nums_to_compare[ideal_topic_num_index]

    ax = sns.lineplot(
        x=topic_nums_to_compare[:-1], y=mean_stabilities, label="Average Topic Overlap"
    )
    ax = sns.lineplot(
        x=topic_nums_to_compare[:-1], y=coherences, label="Topic Coherence"
    )

    ax.axvline(x=ideal_topic_num, label="Ideal Number of Topics", color="black")
    ax.axvspan(
        xmin=ideal_topic_num - 1, xmax=ideal_topic_num + 1, alpha=0.5, facecolor="grey"
    )

    # Set plot limits
    y_max = max(max(mean_stabilities), max(coherences)) + (
        0.10 * max(max(mean_stabilities), max(coherences))
    )
    ax.set_ylim([0, y_max])
    ax.set_xlim([topic_nums_to_compare[0], topic_nums_to_compare[-1] - 1])

    ax.axes.set_title("Method Metrics per Number of Topics", fontsize=25)
    ax.set_ylabel("Metric Level", fontsize=20)
    ax.set_xlabel("Number of Topics", fontsize=20)
    plt.legend(fontsize=20)

    return ax

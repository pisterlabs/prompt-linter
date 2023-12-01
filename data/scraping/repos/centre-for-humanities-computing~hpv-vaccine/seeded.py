'''Iterate guidedlda's LDA in search of good hyperparameters.

TODO
- iterate seed_confidence?
'''
import os
from itertools import chain
from time import time

import ndjson
import numpy as np
from joblib import dump

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import guidedlda
import pyLDAvis.sklearn

from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel

from src.utility.general import make_folders


def init_guidedlda(texts, seed_topic_list):
    '''
    Prepare data & transform seeds to priors for guidedlda.
    (Vectorize texts and extract hashed seeds)

    No more preprocessing is done here.


    Parameters
    ----------
    texts : iterable
        already preprocessed text data you want to build seeds on.

    seed_topic_list : list (of lists)
        list of words, where in seed_topic_list[x][y] 
        x is a topic and y a word belonging in that topic.
    '''
    # texts are already preprocessed, so vectorizer gets this as tokenizer
    def do_nothing(doc):
        return doc

    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=do_nothing,
        preprocessor=None,
        lowercase=False
    )

    # prep texts to guidedlda format
    X = vectorizer.fit_transform(texts)
    tf_feature_names = vectorizer.get_feature_names()
    word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))

    # catch seed-word hashes to give to the model
    seed_priors = {}
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            try:
                seed_priors[word2id[word]] = t_id
            except KeyError:
                pass

    return X, seed_priors, vectorizer


def gensim_format(texts):
    '''
    Tokenized texts to gensim objects.
    For calculating topic coherence using gensim's CoherenceModel.
    '''

    dictionary = corpora.Dictionary(texts)
    bows = [dictionary.doc2bow(tl) for tl in texts]

    return bows, dictionary


def coherence_guidedlda(topics, bows, dictionary, texts):
    '''
    Parameters
    ----------
    topics : list of list of str
        List of tokenized topics.
        Be careful to keep topics in the same order 
        as in your guidedlda model.

    bows : iterable of list of (int, number)
        Corpus in BoW format.

    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        Gensim dictionary mapping of id word to create corpus.

    texts : iterable
        already preprocessed text data you want to build seeds on.

    Returns
    -------
    coh_score : float
        average topics coherence of the whole model

    coh_topics : list 
        per-topic coherence scores (same order as topics)
    '''

    cm = CoherenceModel(
        topics=topics,
        corpus=bows,
        dictionary=dictionary,
        texts=texts,
        coherence='c_v'
    )

    coh_score = cm.get_coherence()
    coh_topics = cm.get_coherence_per_topic()

    return coh_score, coh_topics


def grid_search_lda_SED(texts, seed_topic_list,
                        n_topics_range, priors_range,
                        out_dir,
                        n_top_words=20,
                        seed_confidence=0.15,
                        iterations=2000,
                        save_doc_top=True,
                        verbose=True):
    '''
    Fit many topic models to pick the most tuned hyperparameters.
    Guidedlda version.

    Each fitted model is saved, filename being in the following format:
    {number of topics}T_{iteration rank}I_.{file extension}


    Parameters
    ----------
    texts : iterable
        already preprocessed text data you want to build seeds on.

    seed_topic_list : list of lists
        list of words, where in seed_topic_list[x][y] 
        x is a topic and y a word belonging in that topic.

    n_topics_range : iterable of int | int
        Number of topics to fit the model with.
        When fitting a single model, :int: is enough.
        Otherwise, input list of ints, a range, or other iterables.

    priors_range : list of tuples
        where every 1st element is alpha, every 2nd is eta. 

    out_dir : str
        path to a directory, where results will be saved (in a child directory).

    n_top_words : int, optional (default: 20)
        when extracting top words associated with each topics, how many to pick?

    seed_confidence : float, optional (default: '0.15')
        When initializing the LDA, where are you on the spectrum
        of sampling from seeds (1), vs. sampling randomly (0)?

    iterations : int, optional (default: 2000)
        maximum number of iterations to fit a topic model with.

    save_doc_top : bool
        save documet-topic matices from models?

    verbose : bool, optional (default: True)
        print progress comments.


    Exports
    -------
    out_dir/report_lines/*
        pickled dict with model information
        (n topics, model coherence, per-topic coherence, hyperparameters)
        
    out_dir/models/*
        gensim objects, where the model is saved.
        
    out_dir/plots/*
        pyLDAvis visualizations of the model
    '''
    # INITIALIZATION
    # prepare foldrs
    make_folders(out_dir)

    # paths
    report_dir = os.path.join(out_dir, "report_lines", "")
    model_dir = os.path.join(out_dir, "models", "")
    plot_dir = os.path.join(out_dir, "plots", "")
    doctop_dir = os.path.join(out_dir, 'doctop_mats', '')

    # if a single model is to be fitted,
    # make sure it can be "iterated"
    if isinstance(n_topics_range, int):
        n_topics_range = [n_topics_range]

    # PREPARE DATA
    # for guidedlda fiting
    X, seed_priors, vectorizer = init_guidedlda(
        texts=texts,
        seed_topic_list=seed_topic_list,
    )

    # for coherence counting
    bows, dictionary = gensim_format(texts)

    # TRAIN MODELS
    i = 0
    for n_top in chain(n_topics_range):

        # iterate over priors
        for alpha_, eta_ in priors_range:

            # track time
            start_time = time() # track time
            # track iterations
            topic_fname = str(n_top) + "T_"
            alpha_fname = str(alpha_).replace('.', '') + 'A_'
            eta_fname = str(eta_).replace('.', '') + 'E_'

            # paths for saving
            filename = topic_fname + alpha_fname + eta_fname + 'seed'
            report_path = os.path.join(report_dir + filename + '.ndjson')
            model_path = os.path.join(model_dir + filename + '.joblib')
            pyldavis_path = os.path.join(plot_dir + filename + '_pyldavis.html')
            doctop_path = os.path.join(doctop_dir + filename + '_mat.ndjson')

            # train model
            model = guidedlda.GuidedLDA(
                n_topics=n_top,
                n_iter=iterations,
                alpha=alpha_, eta=eta_,
                random_state=7, refresh=10
            )

            # TODO: iterate seed_confidence?
            model.fit(X, seed_topics=seed_priors, seed_confidence=seed_confidence)

            # track time usage
            training_time = time() - start_time
            if verbose:
                print('    Time: {}'.format(training_time))

            # save priors
            alpha = model.alpha
            eta = model.eta

            # extract topic words
            topics = []
            for i, topic_dist in enumerate(model.topic_word_):
                topic_words = (
                    # take vocab (list of tokens in order)
                    np.array(vectorizer.get_feature_names())
                    # take term-topic distribution (topic_dist),
                    # where topic_dist[0] is probability of vocab[0] in that topic
                    # and sort vocab in descending order
                    [np.argsort(topic_dist)]
                    # selected & reorder so that only words only n_top_words+1 are kept
                    [:-(n_top_words+1):-1]
                )
                # array to list
                topic_words = [word for word in topic_words]
                topics.append(topic_words)

            # calculate topic coherence based on the extracted topics
            coh_score, coh_topics = coherence_guidedlda(
                topics=topics,
                bows=bows,
                dictionary=dictionary,
                texts=texts
            )

            # save report
            report = (n_top, alpha, eta, training_time, coh_score, coh_topics)
            with open(report_path, 'w') as f:
                ndjson.dump(report, f)

            # save model
            dump(model, model_path)

            # produce a visualization
            nice = pyLDAvis.sklearn.prepare(model, X, vectorizer)
            pyLDAvis.save_html(nice, pyldavis_path)

            # save document-topic matrix
            if save_doc_top:
                doc_topic = (
                    model
                    .transform(X)
                    .tolist()
                )

                with open(doctop_path, 'w') as f:
                    ndjson.dump(doc_topic, f)

    return None

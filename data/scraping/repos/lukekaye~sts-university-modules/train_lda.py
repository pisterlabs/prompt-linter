# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from octis.evaluation_metrics.coherence_metrics import Coherence
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pickle

def main():
    '''
    Trains Latent Dirichlet Allocation (LDA) models and saves to ../models
    Uses Topic Coherence (NPMI) to select number of topics in LDA model
    '''
    logger = logging.getLogger(__name__)
    logger.info('training Latent Dirichlet Allocation model from training data ../data/processed/train.pkl')

    # suppress INFO flags that cause spam during Topic Coherence evaluation
    logging.getLogger('gensim.topic_coherence.text_analysis').setLevel(logging.WARNING)
    logging.getLogger('gensim.corpora.dictionary').setLevel(logging.WARNING)
    logging.getLogger('gensim.utils').setLevel(logging.WARNING)
    logging.getLogger('gensim.topic_coherence.probability_estimation').setLevel(logging.WARNING)

    # load train.pkl
    train_path = project_dir.joinpath('data/processed/train.pkl')
    train = pd.read_pickle(train_path)
    train_list = train['Concatenated'].tolist()

    # get the tokenised corpus in list of lists form, only retaining alphanumeric characters
    # only retaining alphanumeric characters should make the NPMI score more accurate
    tokenised_corpus = train['Concatenated'].apply(RegexpTokenizer(r'\w+').tokenize).tolist()

    # load and fit bag-of-words vectoriser
    vectoriser = CountVectorizer(min_df = 2, stop_words = 'english')
    train_vectorised = vectoriser.fit_transform(train_list)
    vocabulary = vectoriser.get_feature_names()

    # find LDA model with highest Topic Coherence; consider 5, 10, ..., 95, 100 topics
    topic_coherence_scores = []
    for num_topics in range(5, 101, 5):
        # fit latent dirichlet allocation model; use perplexity to regulate stopping
        logger.info(f'fitting LDA model with {num_topics} topics and evaluating Topic Coherence (NPMI)')
        lda = LatentDirichletAllocation(n_components = num_topics,
                                        random_state = 1,
                                        verbose = 1,
                                        max_iter = 1000,
                                        evaluate_every = 5)
        lda.fit(train_vectorised)

        # get top-10 most significant words, per topic, in form required by octis for evaluation
        significant_words = []
        for component in lda.components_:
            word_ids = np.argsort(component)[::-1][:10]
            # store the words most relevant to the topic
            significant_words.append([vocabulary[word] for word in word_ids])
        significant_words = {'topics': significant_words}

        # calculate Topic Coherence (NPMI) for LDA model
        topic_coherence_npmi = Coherence(texts = tokenised_corpus, topk = 10, measure = 'c_npmi')
        topic_coherence_npmi_score = topic_coherence_npmi.score(significant_words)
        topic_coherence_scores.append(topic_coherence_npmi_score)
        logger.info(f'LDA model with {num_topics} topics Topic Coherence (NPMI): {topic_coherence_npmi_score}')

    # refit LDA model with largest Topic Coherence (NPMI)
    num_topics_best_lda_model = (topic_coherence_scores.index(max(topic_coherence_scores)) + 1) * 5
    logger.info(
        f'refitting LDA model with largest Topic Coherence (NPMI): {num_topics_best_lda_model} topics, {max(topic_coherence_scores)} NPMI')
    lda_best = LatentDirichletAllocation(n_components = num_topics_best_lda_model,
                                         random_state = 1,
                                         verbose = 1,
                                         max_iter = 1000,
                                         evaluate_every = 5)
    lda_best.fit(train_vectorised)

    # refit LDA model with 80 topics (similar number to most BERTopic models)
    logger.info(f'refitting LDA model with 80 topics')
    lda_80 = LatentDirichletAllocation(n_components = 80,
                                       random_state = 1,
                                       verbose = 1,
                                       max_iter = 1000,
                                       evaluate_every = 5)
    lda_80.fit(train_vectorised)

    # save LDA models
    lda_output = project_dir.joinpath('models/lda_models.pkl')
    with open(lda_output, "wb") as output:
        pickle.dump({'best_lda_model': lda_best,
                     'lda_model_80_topics': lda_80},
                    output,
                    protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('finished training LDA models; output saved as ../models/lda_models.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
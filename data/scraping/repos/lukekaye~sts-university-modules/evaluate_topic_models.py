# -*- coding: utf-8 -*-
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pandas as pd
import pickle
from bertopic import BERTopic
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def get_diversity_and_coherence(significant_words, tokenised_corpus):
    '''
    Return the Topic Diversity and Topic Coherence (NPMI) for a given set of sets of words and a tokenised corpus
    '''
    # find Topic Diversity
    topic_diversity = TopicDiversity(topk = 10)
    topic_diversity_score = round(topic_diversity.score(significant_words), 3)

    # find Topic Coherence
    topic_coherence = Coherence(texts = tokenised_corpus, topk = 10, measure = 'c_npmi')
    topic_coherence_score = round(topic_coherence.score(significant_words), 3)

    return topic_diversity_score, topic_coherence_score


def evaluate_lda_model(model_name, tokenised_corpus, in_path, vocabulary):
    '''
    Evaluate the given fitted LDA model, specifically its Topic Diversity and Topic Coherence
    Topic Coherence is given by Normalised Pointwise Mutual Information (NPMI)
    '''
    # load LDA model
    model_path = project_dir.joinpath(in_path)
    with open(model_path, "rb") as model_input:
        saved_lda_models = pickle.load(model_input)
        lda = saved_lda_models['best_lda_model']

    # get top-10 most significant words per topic
    significant_words = []
    for component in lda.components_:
        word_ids = np.argsort(component)[::-1][:10]
        # store the words most relevant to the topic
        significant_words.append([vocabulary[word] for word in word_ids])
    significant_words = {'topics': significant_words}

    topic_diversity_score, topic_coherence_score = get_diversity_and_coherence(significant_words, tokenised_corpus)

    return [model_name, topic_diversity_score, topic_coherence_score]


def evaluate_topic_model(model_name, documents, tokenised_corpus, in_path, embeddings):
    '''
    Evaluate the given fitted BERTopic model, specifically its Topic Diversity and Topic Coherence
    Topic Coherence is given by Normalised Pointwise Mutual Information (NPMI)
    '''
    # load BERTopic model
    model_path = project_dir.joinpath(in_path)
    model = BERTopic.load(model_path)

    # transform training data by fitted SimCSE BERTopic model
    topics, probs = model.transform(documents, embeddings)

    # get top-10 most significant words per topic
    significant_words = model.get_topic_info()['Representation']

    # drop outlier topic (topic -1, first row)
    significant_words = significant_words.iloc[1:]

    # convert significant_words to list of lists and make the value of a dictionary
    significant_words = significant_words.tolist()
    significant_words = {'topics': significant_words}

    topic_diversity_score, topic_coherence_score = get_diversity_and_coherence(significant_words, tokenised_corpus)

    return [model_name, topic_diversity_score, topic_coherence_score]


def main():
    '''
    Evaluate the Topic Diversity and Topic Coherence of all topic models
    '''
    logger = logging.getLogger(__name__)
    logger.info('evaluating topic models (Topic Diversity & Topic Coherence)')

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
    train_vocabulary = vectoriser.get_feature_names()

    # load training data document embeddings
    train_embeddings_path = project_dir.joinpath('data/processed/train_document_embeddings.pkl')
    with open(train_embeddings_path, "rb") as embeddings_input:
        saved_embeddings = pickle.load(embeddings_input)
        train_longformer_embeddings = saved_embeddings['train_longformer_embeddings']
        train_bigbird_embeddings = saved_embeddings['train_bigbird_embeddings']
        train_distilroberta_embeddings = saved_embeddings['train_distilroberta_embeddings']
        train_all_distilroberta_embeddings = saved_embeddings['train_all_distilroberta_embeddings']
        train_longformer_simcse_embeddings = saved_embeddings['train_longformer_simcse_embeddings']
        train_longformer_ct_embeddings = saved_embeddings['train_longformer_ct_embeddings']
        train_bigbird_simcse_embeddings = saved_embeddings['train_bigbird_simcse_embeddings']
        train_bigbird_ct_embeddings = saved_embeddings['train_bigbird_ct_embeddings']
        train_bigbird_tsdae_embeddings = saved_embeddings['train_bigbird_tsdae_embeddings']
        train_distilroberta_simcse_embeddings = saved_embeddings['train_distilroberta_simcse_embeddings']
        train_distilroberta_ct_embeddings = saved_embeddings['train_distilroberta_ct_embeddings']
        train_distilroberta_tsdae_embeddings = saved_embeddings['train_distilroberta_tsdae_embeddings']
        train_all_distilroberta_simcse_embeddings = saved_embeddings['train_all_distilroberta_simcse_embeddings']
        train_all_distilroberta_ct_embeddings = saved_embeddings['train_all_distilroberta_ct_embeddings']
        train_all_distilroberta_tsdae_embeddings = saved_embeddings['train_all_distilroberta_tsdae_embeddings']

    # create DataFrame to store evaluation metric scores
    topic_metrics = pd.DataFrame(columns = ['Model', 'Topic Diversity', 'Topic Coherence (NPMI)'])
    topic_scores = []


    # evaluate topic models

    # LDA
    topic_scores.append(evaluate_lda_model('LDA_(45_Topics)',
                                           tokenised_corpus,
                                           'models/lda_models.pkl',
                                           train_vocabulary))

    # Longformer
    topic_scores.append(evaluate_topic_model('Longformer-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/longformer-bertopic',
                                             train_longformer_embeddings))
    # BigBird
    topic_scores.append(evaluate_topic_model('BigBird-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/bigbird-bertopic',
                                             train_bigbird_embeddings))
    # DistilRoBERTa
    topic_scores.append(evaluate_topic_model('DistilRoBERTa-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/distilroberta-bertopic',
                                             train_distilroberta_embeddings))
    # all_DistilRoBERTa
    topic_scores.append(evaluate_topic_model('all_DistilRoBERTa-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/all_distilroberta-bertopic',
                                             train_all_distilroberta_embeddings))

    # Longformer-SimCSE
    topic_scores.append(evaluate_topic_model('Longformer-SimCSE-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/longformer-simcse-bertopic',
                                             train_longformer_simcse_embeddings))
    # Longformer-CT
    topic_scores.append(evaluate_topic_model('Longformer-CT-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/longformer-ct-bertopic',
                                             train_longformer_ct_embeddings))

    # BigBird-SimCSE
    topic_scores.append(evaluate_topic_model('BigBird-SimCSE-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/bigbird-simcse-bertopic',
                                             train_bigbird_simcse_embeddings))
    # BigBird-CT
    topic_scores.append(evaluate_topic_model('BigBird-CT-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/bigbird-ct-bertopic',
                                             train_bigbird_ct_embeddings))
    # BigBird-TSDAE
    topic_scores.append(evaluate_topic_model('BigBird-TSDAE-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/bigbird-tsdae-bertopic',
                                             train_bigbird_tsdae_embeddings))

    # DistilRoBERTa-SimCSE
    topic_scores.append(evaluate_topic_model('DistilRoBERTa-SimCSE-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/distilroberta-simcse-bertopic',
                                             train_distilroberta_simcse_embeddings))
    # DistilRoBERTa-CT
    topic_scores.append(evaluate_topic_model('DistilRoBERTa-CT-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/distilroberta-ct-bertopic',
                                             train_distilroberta_ct_embeddings))
    # DistilRoBERTa-TSDAE
    topic_scores.append(evaluate_topic_model('DistilRoBERTa-TSDAE-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/distilroberta-tsdae-bertopic',
                                             train_distilroberta_tsdae_embeddings))

    # all_DistilRoBERTa-SimCSE
    topic_scores.append(evaluate_topic_model('all_DistilRoBERTa-SimCSE-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/all_distilroberta-simcse-bertopic',
                                             train_all_distilroberta_simcse_embeddings))
    # all_DistilRoBERTa-CT
    topic_scores.append(evaluate_topic_model('all_DistilRoBERTa-CT-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/all_distilroberta-ct-bertopic',
                                             train_all_distilroberta_ct_embeddings))
    # all_DistilRoBERTa-TSDAE
    topic_scores.append(evaluate_topic_model('all_DistilRoBERTa-TSDAE-BERTopic',
                                             train['Concatenated'],
                                             tokenised_corpus,
                                             'models/all_distilroberta-tsdae-bertopic',
                                             train_all_distilroberta_tsdae_embeddings))


    # append topic model scores to DataFrame
    for scores in topic_scores:
        topic_metrics.loc[len(topic_metrics)] = scores

    # save evaluation metric scores to file
    topic_metrics_path = project_dir.joinpath('reports/scores/topic_evaluation_scores.csv')
    topic_metrics.to_csv(topic_metrics_path, sep=';', index=False)

    logger.info('finished evaluating topic models, '
                'output saved to ../reports/scores/ as topic_evaluation_scores.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # resolve project directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env by walking up directories until it's found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
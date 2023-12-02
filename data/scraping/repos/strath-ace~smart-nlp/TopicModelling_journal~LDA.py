# This Source Code Form is subject to the terms of the Mozilla Public ---------------------
# License, v. 2.0. If a copy of the MPL was not distributed with this ---------------------
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */ -----------------------------
# ---------------- Copyright (C) 2021 University of Strathclyde and Author ----------------
# -------------------------------- Author: Audrey Berquand --------------------------------
# ------------------------- e-mail: audrey.berquand@strath.ac.uk --------------------------

'''
LDA.py : Train an unsupervised LDA model with the Gensim Python library.
Radim Rehurek and Petr Sojka. “Software Framework for Topic Modelling with Large Corpora”.
English. In: Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks.
http://is.muni.cz/publication/884893/en. Valletta, Malta: ELRA, May 2010, pp. 45–50.

Possibility to run an optimisation process (set opti to True, line 208 of main_LDA) to find the optimal number of topics
for a training corpus. If you prefer to play topics roulette, or have read our paper and seen our suggested topics numbers,
then opti has been set to False by default and you don't need to change anything.

(The User needs to provide the number of topics, path to corpus and model names in main())
'''

import math
import re, os, time
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim # don't remove
import json
import numpy as np
import itertools

from os import listdir
from os.path import isfile
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import KFold

from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.corpora import MmCorpus
from gensim.test.utils import get_tmpfile
from nltk import FreqDist

fileDir = os.path.dirname(os.path.abspath(__file__))  #
parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

'''---------------------------
---------- FUNCTIONS ---------
---------------------------'''

def corpusInsight(path):
    '''
        Displays corpus statistics
        :param corpus path
    '''

    print('Corpus Insight from', os.path.abspath(path), '\n')
    documents = [f for f in listdir(parentDir+path) if isfile(parentDir+path+f)]

    allCorpusTokensPerSentence=[]
    tokensPerDoc=[]

    for d in documents:
        with open(parentDir+path+ d, 'r') as infile:
            input = json.load(infile)
        tokensPerDoc.append(list(itertools.chain(*input)))
        # retrieve tokens per sentence
        for item in input:
            allCorpusTokensPerSentence.append(item)


    # Join tokenized sentences
    allTokens = list(itertools.chain(*allCorpusTokensPerSentence))
    print('Number of documents:', len(tokensPerDoc))
    print('Number of tokens:', len(allTokens))
    dic = set(allTokens)
    print('Initial Dictionary Size:', len(dic))
    print('Check Seed words, not found:')
    S=['thermal', 'thermal_control', 'thermal_control_system', 'heat', 'temperature', 'radiator', 'insulation', 'cooling', 'thermal', 'heating', 'degree', 'thermodynamics', 'multi_layer_insulation', 'coating', 'mirror', 'heater', 'reflector', 'propulsion', 'propulsion_system', 'spacecraft_propulsion', 'propellant_mass', 'delta_v', 'thruster', 'engine', 'propellant', 'ion', 'plasma', 'electric', 'thrust', 'fuel', 'isp', 'impulse', 'power', 'battery', 'cell', 'solar_cell', 'voltage', 'watt', 'current', 'charge', 'discharge', 'power_supply', 'battery_powered', 'primary', 'secondary', 'circuit', 'energy', 'cycle', 'panel', 'efficiency', 'capacity', 'satellite_communication', 'communication', 'band', 'bandwidth', 'packet', 'x_band', 'transmitter', 'receiver', 'frequency', 'antenna', 'relay', 's_band', 'telemetry', 'tracking', 'telecommand', 'reception', 'command', 'network', 'loss', 'signal', 'range', 'wavelength', 'modulation', 'attitude', 'attitude_control', 'guidance', 'navigation', 'momentum', 'angular', 'body', 'freedom', 'wheel', 'motion', 'torque', 'star_tracker', 'sensor', 'inertial', 'coordinate', 'frame', 'axis', 'data_handling', 'data_rate', 'memory', 'storage', 'sram', 'gbit', 'data', 'downlink', 'uplink', 'computer', 'bit', 'measurement', 'execution', 'operation', 'processor', 'environment', 'radiation', 'gamma_radiation', 'particle', 'shield', 'dose', 'ray', 'shielding', 'electron', 'van_allen', 'star', 'solar_wind', 'wind', 'belt', 'protection', 'cosmic', 'debris', 'background']

    for item in S:
        if item not in dic:
            print(item)


    # Show 50 most frequent words
    fdist1 = FreqDist(allTokens)
    allWordsWithFrequency = [[word, frequency] for (word, frequency) in fdist1.most_common(20)]
    print('\n \n Most Common Corpus words:', *allWordsWithFrequency, sep=' \n')

    # Optional - perform tf-idf analysis to rank low informative words and manually add them to stop word list
    #tf_idf(allTokens)

    return

def ldaModelGeneration(train_corpus, test_corpus, topic_number, value_save_model, model_name):
    '''
    LDA model training, visualisation and evaluation

    Inputs:
    - train_corpus: 80% of the corpus to be used for training of LDA model
    - test_corpus: 20% of the corpus to be used for final evaluation of trained model(perplexity+coherence)
    - topic_number: number of latent topics to be found by model
    - value_save_model: if True, will save model, model dictionary and pyldavis visualisation

    Output:
    - perplexity value: evaluation of perplexity of trained model over unseen documents (test_corpus) --> common LDA
    evaluation metrics
    - coherence value: coherence score of trained model over unseen documents (test_corpus) --> less reliable
    '''

    fileDir = os.path.dirname(os.path.abspath(__file__))  #
    parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

    # ------------------------------------------------------------------------------------------------------------
    # LDA MODEL - GENERATION/VISUALISATION WITH TRAINING CORPUS
    # ------------------------------------------------------------------------------------------------------------

    # Create model dictionary
    dictionary = corpora.Dictionary(train_corpus)
    dictionary.filter_extremes(no_below=3)
    print('\n LDA Model Inputs:\n Dictionary Size:', dictionary)

    # Create and save Document-Term matrix
    corpus = [dictionary.doc2bow(tokens) for tokens in train_corpus]
    MmCorpus.serialize(parentDir + '/LDAmodels/reports_unsupervised/corpus_' + str(model_name) + '.mm', corpus)

    # Generate LDA model
    print(len(corpus))
    exit()

    ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=topic_number, passes=500)

    if value_save_model == True:

        # Visualise topics: words and their weights
        print("LDA Topics:")
        for i in ldamodel.show_topics(formatted=False, num_topics=ldamodel.num_topics, num_words=20):
            print(i)

        # Save model
        dictionary.save(parentDir + '/LDAmodels/reports_unsupervised/dic_' + str(model_name) + '.dict')
        ldamodel.save(parentDir+'/LDAmodels/reports_unsupervised/'+str(model_name))
        print('LDA model generated and saved')

        # Save pyldavis (usually takes a few minutes to generate)
        #vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics = False)
        #pyLDAvis.save_html(vis, parentDir + '/LDAmodels/new_unsupervised/LDA_Visualization_' + str(topic_number)+'.html')

    # ------------------------------------------------------------------------------------------------------------
    #                                LDA MODEL - EVALUATION
    # ------------------------------------------------------------------------------------------------------------

    # Use same dictionary as the model was trained with to transform unseen data into Document-Term matrix
    corpusTest = [dictionary.doc2bow(tokens) for tokens in test_corpus]

    # Model Perplexity - must be minimised
    perplexity = ldamodel.log_perplexity(corpusTest)
    perplexityExp = math.exp(perplexity)

    # Topic Coherence
    cm = CoherenceModel(model=ldamodel, corpus=corpusTest, coherence='u_mass')
    coherence = cm.get_coherence()  # get coherence value

    return perplexityExp, coherence

def main_LDA(inputPath, model_name, topic_number):
    '''
    main_LDA: either optimise topic number for given corpus or select a topic number and train/evaluate/save an
    unsupervised LDA model

    The User will need to indicate whether he/she wants to run the 5 cv optimisation, switch parameter opti to True or
    False. If False, also indicates the number of topics the model should find within corpus, by replacing the value of
    parameter best_topic_number
    '''

    start = time.time()
    fileDir = os.path.dirname(os.path.abspath(__file__))  #
    parentDir = os.path.dirname(fileDir)  # Directory of the Module directory

    # --------------------------------------- LOAD, PREPROCESS CORPUS -------------------------------------------
    # Input Corpus Directory - parsed wikipedia pages in json format, based on Corpora/wikiURLList
    # The code to identify and scrap all hyperlinks within a Wikipedia Page with the Python Selenium is available on demand
    filepath = parentDir + inputPath

    # Load pre-processed corpus - list of tokenized document
    documents = [f for f in listdir(filepath) if isfile(filepath + f)]
    doc_preprocessed = []

    print(len(documents), 'documents found in', os.path.abspath(filepath), '\n')

    corpusInsight(inputPath)

    for d in documents:
        with open(filepath + d, 'r') as infile:
            input = json.load(infile)
        doc_preprocessed.append(list(itertools.chain(*input)))
    # ----------------------------------- SEPARATE TRAINING AND TEST SET -----------------------------------------
    # Divide Corpus between training and test set (80/20%) with train_test_split method
    # from sklearn, splits arrays into random train and test subsets
    corpus_train, corpus_test = train_test_split(doc_preprocessed, test_size=0.2)

    # ----------------------------------- LDA OPTIMISATION ON TRAINING SET ----------------------------------------
    # opti = True, will launch the optimisation process (based on 5 fold cross validation) to find the optimal number
    # of topics for the corpus. Might take some time to run.
    # opti = False, the user provides a topic number and by-pass the optimisation process.

    # !!! USER INPUT !!!
    opti = False

    if opti == True:
        # Optimise an lda model over the training corpus with 5 fold cross validation
        # We are looking for the topics number which will minimise perplexity
        # The number of latent topics to be found by the LDA model is a key parameter of the Topic Modeling

        # Topics Number range to be tested
        topic_number_range = list(range(4, 44, 4))

        # For each topics number, we run a 5-fold cross validation
        # This means that for each topics number, we get 5 perplexity and coherence score. These numbers are averaged to get the
        # evaluation of one topics number. The perplexity variance is also saved.
        LDAevaluation = []

        for topic_number in topic_number_range:

            # Choose LDA model name for this iteration
            model_name = 'model_' + str(topic_number)
            progress = (topic_number_range.index(topic_number) + 1) / len(topic_number_range) * 100
            print('Topic Number:', topic_number, ' Progress [%]: ', progress)
            # generate n-fold distributions
            cv = KFold(n_splits=5, shuffle=True)

            # run n lda model for n fold, each time get perplexity + coherence scores.
            fold_perplexity = []
            fold_perplexityExp = []
            fold_coherence = []

            for train_index, test_index in cv.split(corpus_train):
                # generate training and test corpus
                train_corpus = [corpus_train[index] for index in train_index]
                test_corpus = [corpus_train[index] for index in test_index]
                # run lda model for each fold
                perplexity, coherence = ldaModelGeneration(train_corpus, test_corpus, topic_number, False, model_name)
                # save output
                fold_perplexity.append(perplexity)
                fold_coherence.append(coherence)

            # get results of the fold for n topic number
            mean_perplexity = np.mean(fold_perplexity)
            var_perplexity = np.var(fold_perplexity)
            mean_coherence = np.mean(fold_coherence)

            LDAevaluation.append([topic_number, mean_perplexity, var_perplexity, mean_coherence])

        # Save results
        f = open(parentDir + '/outputs/outUnsupervised/newOptimisation/LDAevaluation_new.txt', mode="w", encoding="utf-8")
        for i in LDAevaluation:
            f.write(str(i))
            f.write('\n')
        f.close()

        # print results
        topicsnum = [x for (x, y, z, v) in LDAevaluation]
        mean_perplexity = [y for (x, y, z, v) in LDAevaluation]
        coherence = [v for (x, y, z, v) in LDAevaluation]

        # perplexity results
        fig1 = plt.figure()
        plt.plot(topicsnum, mean_perplexity)
        plt.ylabel('Average Perplexity')
        #plt.grid()
        plt.xlabel('Topics Number')
        fig1.savefig(parentDir+'/outputs/outUnsupervised/newOptimisation/mean_perplexity.png')
        plt.close

         # topic coherence results
        fig3 = plt.figure()
        plt.plot(topicsnum, coherence)
        plt.ylabel('Coherence')
        #plt.grid()
        plt.xlabel('Topics Number')
        fig3.savefig(parentDir+'/outputs/outUnsupervised/newOptimisation/coherence.png')
        plt.close

        # ---------------------------------------------- SELECTION OF BEST MODEL ---------------------------------------
        # The optimal number of topic should get the exponentielle of perplexity closest to 0. To avoid over-fitting, it
        # is however recommended to balance this choice with a pyldavis visualisation which will allow to visualise the
        # topics distribution.
        # Usually the User will select the best topic number suggested by this optimisation process, and re-run the model
        # generation to include the Final Testing (opti = False).

        # get min perplexity
        min_p = min(mean_perplexity)

        # find index of min perplexity, and get topic number
        best_topic_number = LDAevaluation[mean_perplexity.index(min_p)][0]
        print('Suggested best topic number range: ', best_topic_number)

    else:

        print('\n Number of Topics proposed by User: ', topic_number)

        # -------------------------------- FINAL TESTING OF BEST MODEL WITH TEST CORPUS --------------------------------
        # run lda model generation with complete training set, save model, test with testing set and display evaluation
        print('\n FINAL MODEL: ')
        perplexity, coherence = ldaModelGeneration(corpus_train, corpus_test, topic_number, True, model_name)
        print('Final Evaluation, perplexity:', perplexity, ', Topic Coherence:', coherence)

    print('Computation Time:', round((time.time() - start) / 60, 2), 'minutes')

    return()

def main():
    # Input training corpus path (JSON format)
    inputPath = '/processedCorpora/Wiki/'

    # Select topic number, from our journal paper we recommend:
    # topic number of 22 for wikipedia corpus
    # topic number of 30 for ESA CDF reports corpus
    # topic number of 24 for books corpus
    topic_number = 30

    # Select the names of the models
    m = ['test_unsupervisedLDA']

    # Run training of unsupervised LDA models
    for item in m:
        model_name = item
        main_LDA(inputPath, model_name, topic_number)

    return

'''---------------------------
------------ MAIN ------------
---------------------------'''
if __name__ == "__main__":
    main()


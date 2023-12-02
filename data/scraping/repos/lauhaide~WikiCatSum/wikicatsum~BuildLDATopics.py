
# Laura Perez
# Adapted from here https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

import torch
import re
import numpy as np
import pandas as pd
from pprint import pprint
import argparse, sys
import random
import os

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import datapath

# spacy for lemmatization
import spacy

# Plotting tools
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'could', 'who', 'would', 'may'])
stop_words.remove('own') # shouldn't be removed

mallet_path = '~/wikigen/code/mallet-2.0.6/bin/mallet'  # UPDATE THIS PATH

from Constants import SNT, EOT, EOP, BLANK

def sent_to_words(sentences):
    """
    :param sentences: list of sentences
    :return:
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



def readInputFile(inputFileName, preprocessed=False):
    """
    TODO:
    Note: If we are processing tgt is fine to consider the whole abstract as a document.
    However, if we were to process the src, each paragraph should be treated as separate
    document!

    :param inputFileName:
    :return:
    """
    print("Composites are given documents (e.g. lead section). (preprocessed={})".format(preprocessed))
    data = []
    f = open(inputFileName, 'r', encoding='utf8')
    for line in f:
        pureLine = line.strip().replace(EOP, BLANK).replace(SNT,BLANK).replace(EOT, BLANK)
        if preprocessed:
            data.append([pureLine.split()])
        else:
            data.append(pureLine)
    print("Nb of text read: {}".format(len(data)))
    return data

def readInputFileSentence(inputFileName, preprocessed=False):
    print("Composites are sentences. (preprocessed={})".format(preprocessed))
    data = []
    f = open(inputFileName, 'r', encoding='utf8')
    for line in f:
        paragraphs = line.strip().split(EOP)
        for paragraph in paragraphs:
            if not preprocessed:
                data.extend(paragraph.strip().split(SNT))
            else:
                for sent in paragraph.strip().split(SNT):
                        data.append(sent.split())

    print("Nb of text read: {}".format(len(data)))
    return data


def readInputFileElements(inputFileName, sentence=True, L=None):
    """
    Prepare to annotate each element (either paragraphs or sentences) with a topic distribution.
    Paragraphs and sentences, each will be a document to annotate.

    ##dont do this anymore:
    But keep these together/related
    because need to distinguish each subset of them belonging to each sample (src/target) element.

    :param inputFileName:
    :return:
    """
    print("Read by elements  (per sentences/paragraphs) in a document, each of these elements is grouped by document.")
    data = []
    f = open(inputFileName, 'r', encoding='utf8')
    for line in f:
        if L is not None:
            line = line[:L] # remove title?
        paragraphs = line.strip().split(EOP)
        for paragraph in paragraphs:
            if sentence:
                for sent in paragraph.split(SNT):
                    data.append(sent)
            else:
                data.append(paragraph)
    return data

def getTopicKeys(lda_mod):
    topickeys = {}
    for i, keyswords in lda_mod.show_topics(-1, formatted=False, num_words=10):
        topickeys[i] = []
        for term, p in keyswords:
            topickeys[i].append((term, round(p, 2)))
    return topickeys


def topicDistrib(phrase, lda_mod, num_topics, useNTtopics=None, topickeys=None):

    def remove_stopwords(texts):
        return [[word for word in doc if word not in stop_words] for doc in texts]

    topic_vector = []

    phrase_lemmatized = remove_stopwords(phrase)

    phrase_bin = [lda_mod.id2word.doc2bow(p) for p in phrase_lemmatized] #returns lists of processed texts
    all_topics = lda_mod.get_document_topics(phrase_bin, per_word_topics=True)
    for (doc_topics, word_topics, phi_values), phph, phphall, phbin in zip(all_topics, phrase_lemmatized, phrase, phrase_bin):
        x = np.array([t[1] for t in doc_topics],  dtype=float)
        if len(doc_topics) <= num_topics:
            x_new = np.zeros(num_topics + 1 ,  dtype=float) # first t topics from model, + OtherTopic
            if useNTtopics:
                max_n_idx = x.argsort()[-useNTtopics:]
            if len(phph)==0 :
                x_new[num_topics] = 1
            else:
                indices = max_n_idx if useNTtopics else range(len(doc_topics))
                for j in indices:
                    x_new[doc_topics[j][0]] = x[j]

        if x.sum().item()<= 0.0:
            print("ERROR!")
            print(doc_topics)
            exit()
        topic_vector.append(x_new)

    return topic_vector


def prepareInference(inputFileName, sentence=True, bigram_mod=None, L=None):

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    data_lemmatized = []
    data = readInputFileElements(inputFileName, sentence=sentence, L=L)
    for example in data:
        exdata = list(sent_to_words([example]))

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized.append(lemmatization(exdata, allowed_postags=['NOUN', 'VERB']))


        # Remove Stop Words, bigrams
        if bigram_mod:
           data_words_nostops = bigram_mod[remove_stopwords(data_lemmatized)]
        else:
           data_words_nostops = remove_stopwords(data_lemmatized)

    return data_words_nostops


def prepare(inputFileName, sentence=False, bigram_mod=None, trigram_mod=None):
    """ Preprocess input corpus for training an LDA model. This takes raw texts and does lemmatisation  """
    if sentence:
        data = readInputFileSentence(inputFileName)
    else:
        data = readInputFile(inputFileName)

    # data should be a list of sentences, call all in once or sent by sent? is also possible
    data_words = list(sent_to_words(data))

    if bigram_mod is None and trigram_mod is None:
        # Build the bigram and trigram models
        bigram = gensim.models.phrases.Phrases(data_words, min_count=5, threshold=100)# higher threshold fewer phrases.
        #trigram = gensim.models.phrases.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        #trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        print(allowed_postags)
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    #data_words_bigrams = make_bigrams(data_words_nostops)
    data_words_bigrams = data_words_nostops

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, vb
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'VERB'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return corpus, id2word, data_lemmatized, bigram_mod


def prepare_preprocessed(inputFileName, sentence=False):
    """ Preprocess input corpus for training an LDA model. This takes already lemmatised text.  """

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in doc if word not in stop_words] for doc in texts]

    if sentence:
        data = readInputFileSentence(inputFileName, True)
    else:
        data = readInputFile(inputFileName, True)


    data_words_nostops = remove_stopwords(data)


    # Create Dictionary
    id2word = corpora.Dictionary(data_words_nostops)

    # Create Corpus
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data]

    return corpus, id2word, data, None #No bigrams in this case


def inference_preprocessed(inputFileName, sentence=False):

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in doc if word not in stop_words] for doc in texts]

    if sentence:
        data = readInputFileSentence(inputFileName, True)
    else:
        data = readInputFile(inputFileName, True)


    data_words_nostops = remove_stopwords(data)


    return data_words_nostops


def compute_coherence_values(args, id2word, corpus, texts, limit, start=2, step=3, libraryMallet=False, validset=None):
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
    f=open(os.path.join(args.outdir,'search.mallet.log' if libraryMallet else 'search.gensim.log'),'w')
    for num_topics in range(start, limit, step):
        print("Creating model {} topics".format(num_topics))
        if libraryMallet:
            model = build_model_mallet(corpus, id2word, num_topics=num_topics)
        else:
            model = build_model_gensim(corpus, id2word, num_topics=num_topics, validset=validset)
        #model_list.append(model)
        if validset:
            valid_corpus, valid_id2word, valid_data_lemmatized = validset
            coherencemodel = CoherenceModel(model=model, texts=valid_data_lemmatized, dictionary=valid_id2word, coherence='c_v')
        else:
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        f.write("Model #{}: {}\n".format(num_topics, coherencemodel.get_coherence()))
        f.flush()
    f.close()
    return model_list, coherence_values


def search_model(args, corpus, id2word, texts, libraryMallet=False, validset=None):

    model_list, coherence_values = compute_coherence_values(
        args, id2word, corpus, texts, limit=100, start=10, step=10, libraryMallet=libraryMallet, validset=validset)

    limit = 100;
    start = 10;
    step = 10;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig('coherence_values.png', bbox_inches='tight')

    best, ibest = 0.0, 0
    for i, (m, cv) in enumerate(zip(x, coherence_values)):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
        if cv > best:
            best, ibest = cv, i

    # Select the model and print the topics *TBC* not working the following piece of code
    #optimal_model = model_list[ibest]
    #model_topics = optimal_model.show_topics(formatted=False)
    #pprint(optimal_model.print_topics(num_words=10))
    #temp_file = "best_{{0:.2f}".format(round(best,2))
    #optimal_model.save(temp_file)
    #print("Best model saved: " + "best_{{0:.2f}".format(round(best,2)))

def build_model_mallet(corpus, id2word, num_topics=20, prefix=None):

    # this call (i.e. with random seed) requires gensim '3.7.1'
    ldamallet = gensim.models.wrappers.ldamallet.LdaMallet(mallet_path,
                                                 corpus=corpus,
                                                 num_topics=num_topics,
                                                 id2word=id2word,
                                                 optimize_interval=10, # let the model do optimisation of hyper-parameters, dirichlet
                                                 iterations=1000,
                                                 random_seed = 1,
                                                 prefix=prefix)

    return ldamallet


from gensim.models.callbacks import PerplexityMetric
from gensim.test.utils import common_corpus
def build_model_gensim(corpus, id2word, num_topics=20, validset=None):
    """
    Suggested parameters for better speed
    lda = models.LdaModel(corpus, num_topics=45, id2word=dictionary, update_every=5, chunksize=10000, passes=1)

    :param corpus:
    :param id2word:
    :param num_topics:
    :return:
    """

    # Build LDA model
    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics,
                                               random_state=100,
                                               eval_every=5,
                                               chunksize=10000, #nb of docs in each training chunk
                                               passes=50,
                                               iterations=500,
                                               alpha=0.001,
                                               per_word_topics=True,
                                               workers=4,)

    print("eta",lda_model.eta)
    print("alpha",lda_model.alpha)

    if validset:
        valid_corpus, valid_id2word, valid_data_lemmatized = validset
        print(lda_model.log_perplexity(valid_corpus, len(valid_corpus)))

    return lda_model


def build_topic_sequence(lemText, wordTopics, id2word):
    topicSeq = []
    topicDict = {}
    topicSet = set()
    for id, t in wordTopics:
      topicDict[id2word[id]] = ",".join([str(x) for x in t])
      topicSet.update(set([x for x in t]))
    for w in lemText:
        if w in topicDict.keys():
            topicSeq.append("{}:{}".format(w, topicDict[w]))
    return " ".join(topicSeq), len(topicSet)


def main():

    usage = 'usage:\npython BuildLDATopics.py --corpus <corpus>' \
            '\n '
    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--corpus', help="Input file containing the text documents.", required=False)
    parser.add_argument('--num-topics', help="Number of topics to train the model for (Use mallet model).", required=False)
    parser.add_argument('--search-num-topics',
                        help="Train models for differnt nb of topics to search for the best coherence score. "
                             "(Use mallet model)", action='store_true', required=False)
    parser.add_argument('--annotate', help="Annotate corpus with LDA model.", required=False)
    parser.add_argument('--loadmodel', help="Load LDA model.", required=False)
    parser.add_argument('--savemodel', help="Save LDA model as.", required=False)
    parser.add_argument('--outdir', help="Directory where to save outputs.")
    parser.add_argument('--mallet', help="Use mallet to build LDA model.", action='store_true',required=False)
    parser.add_argument('--sent-level', help="composites are sentences.", action='store_true', required=False, default=False)
    parser.add_argument('--bowpreproc', help="the given input file is the preprocessed input text", action='store_true', required=False,
                        default=False)


    args = parser.parse_args(sys.argv[1:])
    np.random.seed(1)
    print("From preprocessed corpus = {}".format(args.bowpreproc))

    num_topics = 20
    if args.num_topics:
        num_topics = int(args.num_topics)

    if args.search_num_topics:
        if args.bowpreproc:
            corpus, id2word, data_lemmatized, _ = prepare_preprocessed(args.corpus, args.sent_level)
            valid_corpus, valid_id2word, valid_data_lemmatized, _ = \
                prepare_preprocessed(args.corpus.replace("train","valid"), args.sent_level)
        else:
            corpus, id2word, data_lemmatized, _ = prepare(args.corpus, args.sent_level)
            valid_corpus, valid_id2word, valid_data_lemmatized, _ = \
                prepare(args.corpus.replace("train","valid"), args.sent_level)
        print("Data is ready!")
        print("*    Search for LDA models")
        search_model(args, corpus, id2word, data_lemmatized, libraryMallet=args.mallet,)
                     #validset=(valid_corpus, valid_id2word, valid_data_lemmatized))

        return

    elif args.annotate:
        print("*    Annotate corpus")
        if args.loadmodel:
            #if args.mallet:
            lda = gensim.models.ldamodel.LdaModel.load(args.loadmodel)
            #else:
            #    lda = gensim.models.ldamodel.LdaModel.load(args.loadmodel)
            bigram_mod = None
            if os.path.isfile(args.loadmodel + "_bigrams"):
                bigram_mod = gensim.models.phrases.Phraser.load(args.loadmodel + "_bigrams")
            print("*    Model is loaded")
            modelName = args.loadmodel[args.loadmodel.rfind("/")+1:]
        else:
            if args.bowpreproc:
                corpus, id2word, data_lemmatized = prepare_preprocessed(args.corpus, args.sent_level)
            else:
                corpus, id2word, data_lemmatized = prepare(args.corpus, args.sent_level)
            print("Data is ready!")
            if args.mallet:
                lda = build_model_gensim(corpus, id2word, num_topics=num_topics)
            else:
                lda = build_model_mallet(corpus, id2word, num_topics=num_topics)
            lda.save(os.path.join(args.outdir, args.savemodel))
            print("*    Model created and saved")
            modelName = args.savemodel

        print("Saving topics...")
        f = open(args.annotate+"_"+modelName+"-topics.txt", 'w', encoding='utf8')
        for i, keyswords in lda.show_topics(-1, formatted=False, num_words=10):
            f.write( "Topic #" + str(i) + ":\n")
            for term, p in keyswords:
                f.write("\t{}, {}\n".format(term, round(p,2)))
            f.write("\n")
        f.close()

        print("Annotating corpus...")
        oriTexts = []
        inf = open(args.annotate, 'r', encoding='utf8')
        for line in inf:
            oriTexts.append(line.strip())
        f = open(args.annotate+"_"+modelName+"-topics_dist.txt", 'w', encoding='utf8')
        validdata_lemmatized = inference_preprocessed(args.annotate, args.sent_level)
        print("size of corpus to annotate: {}".format(len(validdata_lemmatized)))

        valid_corpus = [lda.id2word.doc2bow(text) for text in validdata_lemmatized]
        print("...loaded")
        vlen = len(valid_corpus)
        if args.mallet:
            for i in range(vlen):
                print(" ".join([str(x[0])+"("+str(x[1])+")  " for x in lda[valid_corpus[i]] if x[1]>0.01]))
                f.write(" ".join([str(x[0]) for x in lda[valid_corpus[i]] if x[1]>0.01]) + '\n')

        else:
            #model is gensim
            all_topics = lda.get_document_topics(valid_corpus, per_word_topics=True)
            cnt = 0
            for doc_topics, word_topics, phi_values in all_topics:
                f.write('\nTarget document:\n')
                f.write(oriTexts[cnt])
                tseq, tset = build_topic_sequence(validdata_lemmatized[cnt][0], word_topics, lda.id2word)
                f.write('\nTopic sequence: \n{}'.format(tseq))
                f.write('\nNb distinct topics: {}\n'.format(tset))
                f.write('\ntopic distrib: {}\n'.format(doc_topics))
                cnt+=1
                print('\ntopic distrib: {}\n'.format(doc_topics))

            y = lda.expElogbeta
            print(y.shape)

            lda.show_topics(-1, len(lda.id2word))

            print(lda.get_term_topics("locate"))

        f.close()

    elif args.mallet and not args.annotate:
        logging.basicConfig(filename=os.path.join(args.outdir, args.savemodel+'.log'),
                            format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        print("*    Build LDA mallet model")
        if args.bowpreproc:
            corpus, id2word, data_lemmatized, bigram_mod = prepare_preprocessed(args.corpus, args.sent_level)
        else:
            corpus, id2word, data_lemmatized, bigram_mod = prepare(args.corpus, args.sent_level)
        print("Data is ready!")
        lda = build_model_mallet(corpus, id2word, num_topics=num_topics, prefix=os.path.join(args.outdir, args.savemodel.replace("mallet","")))
        print("saving models...")
        lda.save(os.path.join(args.outdir, args.savemodel))
        if bigram_mod:
            bigram_mod.save(os.path.join(args.outdir, args.savemodel + "_bigrams"))

        #check files can be loaded
        loadedOK = gensim.models.wrappers.LdaMallet.load(os.path.join(args.outdir, args.savemodel))
        print("State file: "+loadedOK.fstate())
        if bigram_mod:
            loadedOK = gensim.models.phrases.Phraser.load(os.path.join(args.outdir, args.savemodel + "_bigrams"))
        print("Saved models ok!")

        pprint(lda.show_topics(formatted=False))

        # Compute Coherence Score
        coherence_model_ldamallet = CoherenceModel(model=lda, texts=data_lemmatized, dictionary=id2word,
                                                   coherence='c_v')
        coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        print('\nCoherence Score: ', coherence_ldamallet)

    else:
        logging.basicConfig(filename=os.path.join(args.outdir, args.savemodel+'.log'),
                            format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        print("*    Build LDA Gensim model")
        if args.bowpreproc:
            corpus, id2word, data_lemmatized, bigram_mod = prepare_preprocessed(args.corpus, args.sent_level)
        else:
            corpus, id2word, data_lemmatized, bigram_mod = prepare(args.corpus, args.sent_level)
        print("Data is ready!")
        lda = build_model_gensim(corpus, id2word, num_topics=num_topics)
        print("saving models...")
        lda.save(os.path.join(args.outdir, args.savemodel))
        if bigram_mod:
            bigram_mod.save(os.path.join(args.outdir, args.savemodel + "_bigrams"))

        #check files can be loaded
        loadedOK = gensim.models.ldamodel.LdaModel.load(os.path.join(args.outdir, args.savemodel))
        if bigram_mod:
            loadedOK = gensim.models.phrases.Phraser.load(os.path.join(args.outdir, args.savemodel + "_bigrams"))
        print("saved models ok!")

        pprint(lda.show_topics(formatted=False))

        # Compute Coherence Score
        coherence_model = CoherenceModel(model=lda, texts=data_lemmatized, dictionary=id2word,
                                                   coherence='c_v')
        coherence_lda = coherence_model.get_coherence()
        print('\nCoherence Score: ', coherence_lda)




if __name__ == '__main__':
    main()

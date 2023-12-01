import argparse
import os
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
from shared import helpers
from shared import db
from shared.models import CorpusIter, BoWIter
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
import parmap

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db-path',
                    type=str,
                    default='../data/news.db',
                    help='the path to database where news articles are stored')

parser.add_argument('-o', '--output-path',
                    type=str,
                    default='../models',
                    help='output path where trained model should be stored')

parser.add_argument('-m', '--month',
                    choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    type=int,
                    default=None,
                    help='The month on which model is to be trained. Do not pass any value, if you want to '
                         'train on the entire year')

parser.add_argument('-y', '--year',
                    choices=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
                             2015, 2016, 2017, 2018],
                    type=int,
                    default=2015,
                    help='The year of data sources on which model is to be trained')

parser.add_argument('-dict', '--dictionary',
                    type=str,
                    default=None,
                    help='the path to vocab. only required training lda')

parser.add_argument('-p', '--phraser',
                    type=str,
                    default='None',
                    help='the path to gensim bigram/phraser model. only required when training lda')

parser.add_argument('--priors',
                    type=str,
                    default=None,
                    help='the path to expected topics bigrams. only required when training lda')

parser.add_argument('--mallet-path',
                    type=str,
                    default='~/Mallet/bin/mallet',
                    help='the path to mallet binary. only required when training lda using mallet implementation')

parser.add_argument('-t', '--type',
                    type=str,
                    choices=['dict', 'lda', 'lda-mallet', 'model-selection'],
                    required=True,
                    help='build the dictionary or train lda')

parser.add_argument('-l', '--limit',
                    type=int,
                    default=50,
                    help='upper limit on number of topics. to be used when performing model selection')

parser.add_argument('-s', '--step',
                    type=int,
                    default=5,
                    help='step by which num topics increase. to be used when performing model selection.')


def create_eta(priors, etadict, ntopics):
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=0.1)  # create a (ntopics, nterms) matrix and fill with 1
    for word, topic in priors.items():  # for each word in the list of priors
        keyindex = [index for index, term in etadict.items() if term == word]  # look up the word in the dictionary

        if len(keyindex) > 0:  # if it's in the dictionary
            eta[topic, keyindex[0]] = 1e7  # put a large number in there
    eta = np.divide(eta, eta.sum(axis=0))  # normalize so that the probabilities sum to 1 over all topics
    return eta


def viz_model(model, modeldict):
    ntopics = model.num_topics
    # top words associated with the resulting topics
    topics = ['Topic {}: {}'.format(t, modeldict[w]) for t in range(ntopics) for w, p in
              model.get_topic_terms(t, topn=1)]
    terms = [modeldict[w] for w in modeldict.keys()]
    fig, ax = plt.subplots()
    ax.imshow(model.get_topics())  # plot the numpy matrix
    ax.set_xticks(modeldict.keys())  # set up the x-axis
    ax.set_xticklabels(terms, rotation=90)
    ax.set_yticks(np.arange(ntopics))  # set up the y-axis
    ax.set_yticklabels(topics)
    plt.savefig('./lda_results.png')


def compute_coherence_values(mallet_path, dictionary, corpus, texts, limit, start=2, step=3):
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
    for num_topics in range(start, limit, step):
        print('lda mallet with topics = {}'.format(num_topics))
        model = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, alpha=0.1,
                                          id2word=dictionary, iterations=500, random_seed=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def plot_coherence_vs_topic_nums(coherence_values, start, limit, step):
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.savefig('./coherence_lda.png')


def build_dictionary_and_bigram_model(docs, out, name):
    print('preprocessing corpus')
    corpus = list(iter(CorpusIter(docs, helpers.preprocess_text_for_lda)))
    print('identifying phrases')
    bigram_model = Phrases(corpus, min_count=1, threshold=0.5)
    print(bigram_model.vocab)
    print('bigram model built!!')
    print('now computing the dictionary')
    dictionary = corpora.Dictionary(bigram_model[corpus])
    dictionary.filter_extremes(no_above=0.40, no_below=3)

    print('saving dictionary & bigram model')
    dictionary.save(os.path.join(out, 'topics_vocab_{}.dict'.format(name)))
    bigram = Phraser(bigram_model)
    bigram.save(os.path.join(out, "bigram_{}.pkl".format(name)))


def train_lda(docs, priors, dictionary, bigram_model, out, name):
    filter_fn = helpers.preprocess_text_for_lda

    print('converting corpus into bag of words')
    bow_articles = list(iter(BoWIter(dictionary, docs, filter_fn, bigram=bigram_model)))
    print('training lda')
    eta = create_eta(priors, dictionary, 20)
    lda_model = models.ldamulticore.LdaMulticore(corpus=bow_articles,
                                                 id2word=dictionary,
                                                 passes=2,
                                                 eta=eta,
                                                 random_state=42,
                                                 per_word_topics=True,
                                                 iterations=100,
                                                 num_topics=20)

    lda_model.save(os.path.join(out, 'lda_model_{}.pkl'.format(name)))
    print(lda_model.print_topics())


def apply_fn(doc, bigram, preprocess_fn):
    return bigram[preprocess_fn(doc.transcript)]


def train_lda_mallet(mallet_path, docs, dictionary, bigram_model, out, name):
    filter_fn = helpers.preprocess_text_for_lda

    print('converting corpus into bag of words')
    bow_articles = parmap.map(apply_fn, docs, dictionary, bigram_model, filter_fn, pm_pbar=True)

    print('training lda using mallet implementation')
    model = models.wrappers.LdaMallet(mallet_path, corpus=bow_articles, num_topics=25, id2word=dictionary)
    lda_model = models.wrappers.ldamallet.malletmodel2ldamodel(model)

    print('saving lda model')
    lda_model.save(os.path.join(out, 'lda_model_{}.pkl'.format(name)))
    print(lda_model.print_topics())


def perform_model_evaluation(mallet_path, dictionary, bigram_model, corpus, limit, start=2, step=3):
    filter_fn = helpers.preprocess_text_for_lda
    print('converting corpus to bag of words')
    X = parmap.map(apply_fn, corpus, bigram_model, filter_fn, pm_pbar=True)
    bow_X = parmap.map(dictionary.doc2bow, X, pm_pbar=True)

    print('performing model selection using coherence scores')
    lda_models, coh_score = compute_coherence_values(mallet_path, dictionary, bow_X, X, limit, start, step)
    plot_coherence_vs_topic_nums(coh_score, start, limit, step)


def main():
    args = parser.parse_args()
    conn = db.NewsDb(args.db_path)

    train = args.type

    if args.month is None:
        corpus = list(conn.select_articles_by_year(args.year))
        out_file_name = '{}'.format(args.year)
    else:
        corpus = list(conn.select_articles_by_year_and_month(args.year, args.month))
        out_file_name = '{}_{}'.format(args.year, args.month)

    if train == 'dict':
        assert args.dictionary is None, "error, dictionary passed with type dict"
        build_dictionary_and_bigram_model(corpus, args.output_path, out_file_name)
    elif train == 'lda':
        assert args.dictionary is not None, "training lda but dictionary not passed"
        assert args.phraser is not None, "training lda but bigram model is not passed"
        priors = helpers.load_json(args.priors)
        dct = corpora.Dictionary.load(args.dictionary)
        bigram_model = models.phrases.Phraser.load(args.phraser)
        train_lda(corpus, priors, dct, bigram_model, args.output_path, out_file_name)
    elif train == "lda-mallet":
        dct = corpora.Dictionary.load(args.dictionary)
        bigram_model = models.phrases.Phraser.load(args.phraser)
        mallet_path = args.mallet_path
        train_lda_mallet(mallet_path, corpus, dct, bigram_model, args.output_path, out_file_name)
    elif train == "model-selection":
        dct = corpora.Dictionary.load(args.dictionary)
        bigram_model = models.phrases.Phraser.load(args.phraser)
        mallet_path = args.mallet_path
        perform_model_evaluation(mallet_path, dct, bigram_model, corpus, args.limit, start=5, step=args.step)


if __name__ == '__main__':
    main()

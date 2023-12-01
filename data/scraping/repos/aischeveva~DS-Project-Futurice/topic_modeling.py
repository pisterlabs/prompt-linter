import re, functools, operator, collections, random
import numpy as np
import pandas as pd
from utils import *
from sklearn.utils import shuffle
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

def get_all_bows(corpus, dic, bigram=None):
    """ Get BoW format for all documents in the given 'corpus'.
        --------------------
        Parameter:
            corpus: all raw text documents
            dic: trained Dictionary
            bigram: gensim.Phraser

        Return:
            list of BoW-formatted documents
    """
    all_bows = []
    for year in corpus:
        bows = []
        for doc in year:
            # Get tokens from documents:
            text = simple_preprocess(doc)
            if bigram:
                text = bigram[text]
            # Convert tokens to BoW format:
            bow = dic.doc2bow(text)
            bows.append(bow)
        # Merge BoWs of 1 year into 1 list of BoW:
        all_bows.append(bows)
    return all_bows

def texts_bows_dict(docs, no_below, no_above, min_count, threshold,
        bigram=True):
    """ Tokenize every documents and get their BoW format.
        Return also
        --------------------
        Parameter:
            docs: document corpus
            no_below: filter words that appear in less than
                      'no_below' number of document
            no_above: filter words that appear in more than
                      'no_above' percent of document.
            min_count : filter words that appear in less than
                      'min_count' times in a document
            threshold: filter words that appear in more than
                      'threshold' times in a document
            bigram: if True, then include bigram in the model
        Return:
            (texts (list of list of tokens),
            bows (list of BoW-formatted documents),
            dictionary,
            bigram)
    """
    # Tokenize documents:
    texts = [simple_preprocess(doc) for doc in docs]

    if bigram:
        tmp = Phrases(texts, min_count=min_count, threshold=threshold)
        bigram = Phraser(tmp)
        texts = [bigram[doc] for doc in texts]

    # Create a dictionary from 'docs' containing
    # the number of times a word appears in the training set:
    dictionary = gensim.corpora.Dictionary(texts)

    # Filter extremes vocabularies:
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    # Create the Bag-of-words model for each document i.e for
    # each document we create a dictionary reporting how many
    # words and how many times those words appear:
    bows = [dictionary.doc2bow(text) for text in texts]

    return texts, bows, dictionary, bigram

def models_codherence_perplexity(texts, bows, dic,
        topic_start=100, topic_end=201, step=10,
        chunk=10, passes=3, cores=2):
    """ Build models on a range of number of topics to compare quality.
        The output is 3 lists of:
            1. List of built models
            2. List of coherence scores calculated on texts
            3. List of perplexity scores calculated on bows
        --------------------
        Parameter:
            texts: list of list of tokens
            bows: list of list of BoWs
            dic: dictionary of id <-> word
            topic_start, topic_end, step: range of number of topics
            chunk: number of data used in each training step
            passes: number of passes through the whole training data
            cores: number of cores use for parallel training

        Return:
            models, coherence_scores, perplexity_scores
    """
    models = []
    coherence_scores = []
    perplexity_scores = []
    for num_topics in range(topic_start, topic_end, step):
        print('Building model of %d topics' % (num_topics))
        # Build topic model for the given number of topics:
        model = LdaMulticore(corpus=bows, id2word=dic,
                             eta='auto', num_topics=num_topics,
                             chunksize=chunk, passes=passes, workers=cores)
        # Build coherence model to test the topic model:
        coherence_model = CoherenceModel(model=model, texts=texts,
                                         dictionary=dic, coherence='c_v')
        # Save the results:
        models.append(model)
        coherence_scores.append(coherence_model.get_coherence())
        perplexity_scores.append(model.log_perplexity(bows))
    return models, coherence_scores, perplexity_scores

def topic_union(top_topics, topic_list, corr, num):
    """ Get a collection of preference topics.
        Preference topics is a union of top topics w.r.t
        coherence score top topics w.r.t independence (least
        correlated with other topics).
        --------------------
        Parameter:
            top_topics: top topics w.r.t cohenrence score
            topic_list: list of all topics in the LDA model
            corr: correlation matrix of all topics in LDA model
            num: number of candidate topics from 2 sources

        Return:
            list of topic ID
    """
    # Get the topic map:
    topic_map = dict(topic_list)
    topic_map = {k: re.findall(r'[a-z_]+', v) for k, v in topic_map.items()}
    topic_map = {''.join(v): k for k, v in topic_map.items()}
    # Get the top independent topics:
    corr_sum = np.sum(corr, axis=1)
    top_independence = []
    for _ in range(num):
        top_index = np.argmax(corr_sum)
        corr_sum[top_index] = 0
        top_independence.append(top_index)
    # Get the top coherence topics:
    top_coherence = [[q[1] for q in p[0]] for p in top_topics[:num]]
    top_coherence = [''.join(presentation) for presentation in top_coherence]
    top_coherence = [topic_map[pre] for pre in top_coherence]
    # Return union of top coherence and top independence topics:
    return sorted(list(set(top_independence).union(set(top_coherence))))

def convert_topic(topics, union, corr):
    """ Convert topics toward union of top topics based on correlation score.
        That is, if a predicted topic in 'topics' is not a member of union,
        then replace this topic by one of the member of union most correlated
        to it.
        --------------------
        Parameter:
            topics: list of topic ID
            union: list of topic ID (top topics)
            corr: correlation matrix of all topics in LDA model

        Return:
            list of topic ID
    """
    for i in range(len(topics)):
        if topics[i] not in union:
            corr_score = corr[topics[i]][union]
            topics[i] = union[np.argmin(corr_score)]
    return topics

def topic_count_years(corpus, model, min_prob, union, corr):
    """ Count the occurrence of predicted topics for the given documents.
        That is,
            1. use 'model' to predict most probable topics for every documents
               in 'bows'
            2. convert any topic that is not a member of 'union'
            3. count the occurrence of every topics in the whole 'bows'
        --------------------
        Parameter:
            corpus: list of list of BoW-formatted documents
            model: LDA model
            min_prob: minimum probability for slecting predicted topics
            union: list of topic ID (top topics)
            corr: correlation matrix of all topics in LDA model

        Return:
            list of list of (topic ID, count)
    """
    counts = []
    for bows in corpus:
        # Predict some most probable topics for documents:
        predicted_topics = [model.get_document_topics(bow,
            minimum_probability=min_prob) for bow in bows]
        # Get only the topic ID:
        predicted_topics = [[p[0] for p in l if p[0] in union]
                for l in predicted_topics]
        # Convert topics that is not in 'union':
        topics = [convert_topic(t, union, corr) for t in predicted_topics]
        # Concatenate all predicted topics into 1 list:
        topics = functools.reduce(operator.iconcat, topics, [])
        # Count each topic:
        count = {topic: 0 for topic in union}
        for topic in topics:
            count[topic] += 1
        count = list(count.items())
        count.sort()
        # Append the topic count of each year to final result:
        counts.append(count)
    return counts

def sampling_corpus(corpus, percent=0.2):
    """ Sample a subset of documents from each year in the 'corpus'.
        --------------------
        Parameter:
            corpus: list of list of BoW-formatted documents
            percent: sampling percentage

        Return:
            list of documents
    """
    sample = []
    num = int(len(corpus[0])*percent)
    for cor in corpus:
        sample = sample + random.choices(cor, k=num)
    return sample

def run(office, sector, same_companies, start_year=2010,
        end_year=2019, use_perplexity=False):
    """ Analyze topics for the 'office' and 'sector'.
        --------------------
        Parameter:
            office: list of list of BoW-formatted documents
            sector: sampling percentage
            same_companies: if True, then only analyze companies that
                    present in every years
            start_year: starting year of interest
            end_year: ending year of interest
            use_perplexity: if True, then also use perplexity score
                    in choosing trained models

        Return:
            None
    """
    # Get the corpus:
    if same_companies:
        corpus = query_intersection(2010, 2019, office, sector, False)
    else:
        corpus = query_docs(2010, 2019, office, sector, False)
    # Sampling documents in each year for training:
    docs = sampling_corpus(corpus, percent=1/(end_year - start_year))
    # Covert documents to tokens, bag of word and dictionary format:
    texts, bows, dic, bigram = texts_bows_dict(docs, 5, 0.5, 5, 100, True)
    # Build models for comparison:
    start = max(len(docs) - 70, 10)
    end = len(docs) + 1
    step = 10
    models, coherences, perplexities = models_codherence_perplexity(
            texts, bows, dic,                            \
            topic_start=start, topic_end=end, step=step, \
            chunk=20, passes=3)
    # Choose a good model:
    if use_perplexity:
        per = [-p for p in perplexities]
        per = [(p - min(per))/(max(per) - min(per)) for p in per]
        score = [per[i]*coherences[i] for i in range(len(per))]
        which = np.argmax(score)
    else:
        which = np.argmax(coherences)
    chosen = models[which]
    # Get texts and bows for each year:
    bows_vs_years = get_all_bows(corpus, dic, bigram)
    # Prepare to get topic union:
    topic_list = chosen.show_topics(chosen.num_topics, 10)
    top_topics = chosen.top_topics(texts=texts, coherence='c_v', topn=10)
    # Get the correlation matrix:
    mdiff, _ = chosen.diff(chosen, distance='jaccard', num_words=100)
    # Get top topics based on coherence and correlation:
    union = topic_union(top_topics, topic_list, mdiff, 10)
    # Get the count for each topic in each year:
    counts = topic_count_years(bows_vs_years, chosen, 0.05, union, mdiff)
    # Get DataFrame:
    data = [[p[1] for p in count] for count in counts]
    pre = [' | '.join(re.findall(r'[a-z_]+', topic_list[i][1])) \
            for i in union]
    df = pd.DataFrame(data, columns=pre, index=range(2010, 2019))
    # Save the model:
    df.to_csv(os.getcwd()[:-14] + '/web/source/' + office + '_' + sector + '.csv')





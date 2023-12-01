from python.ConfigUser import path_data, path_project
from python.params import params as p
import spacy, pandas, traceback, os, time
import numpy as np
import pprint as pp
from python._ProcessingFunctions import MakeListInLists, FlattenList
from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel
from gensim.matutils import jaccard, hellinger
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import datapath
import matplotlib.pyplot as plt


def Load_Sentiment_final():
    """
    Reads in final SentimentList_final from 02_ProcessSentimentlists.py.
    This will be our final dictionary of phrases with sentiments.
    Load it and prepare for GetSentiments()

    :return: Pandas dataframe with phrs and sentiments
    """
    # Read in final Sentimentlist
    df_sentiment_final = pandas.read_csv(path_data + 'Sentiment/Sentimentlist_final.csv', sep=',')

    # convert all words to lower case
    df_sentiment_final['phrase'] = [i.lower() for i in df_sentiment_final['phrase']]

    df_sentiment_final['phrase_sorted'] = df_sentiment_final['phrase'].apply(lambda x: ' '.join(sorted(x.split())))
    print('Sentiment final list loaded')

    return df_sentiment_final


def Load_SePL(type='default'):
    """
    Reads in SePL file, prepares phrases and sorts them; this is required be be run before MakeCandidates() and
    GetSentiments()

    :param type: default 'default' (calls SePL_v1.1.csv), 'modified' (calls SePL_v1.3_negated_modified.csv)
    :return: Pandas dataframe with sepl-phrs and sentiments
    """
    if type == 'modified':
        # Read in modified SePL
        df_sepl = pandas.read_csv(path_data + 'Sentiment/SePL/SePL_v1.3_negated_modified.csv', sep=';')
    else:
        # Read in default SePL
        df_sepl = pandas.read_csv(path_data + 'Sentiment/SePL/SePL_v1.1.csv', sep=';')

    # convert all words to lower case
    df_sepl['phrase'] = [i.lower() for i in df_sepl['phrase']]

    df_sepl['phrase_sorted'] = df_sepl['phrase'].apply(lambda x: ' '.join(sorted(x.split())))
    print('SePL ({}) file loaded'.format(type))

    return df_sepl


nlp2 = spacy.load('de_core_news_md', disable=['ner', 'parser'])


def MakeCandidates(sent, df_sepl=None, get='candidates', verbose=False, negation_list=None):
    """
    prepares a nested list of candidates, make sure df_sepl is loaded (run Load_SePL() before)
    :param sent: input is a full sentence as string
    :param df_sepl: load df_sepl via Load_SePL()
    :param verbose: display
    :param get: default 'candidates', else specify 'negation' to geht same list in lists but only negation words
    :param negation_list: specifiy list with negation words to identify negated sentences; negation_list must not specified
    :return: nested list of lists where each nested list is separated by the POS tag $,
    """

    sent = sent.split(',')
    sent = [nlp2(s) for s in sent]
    candidates = []

    if negation_list is None:
        # Rill (2016)
        # negation_list = ['nicht', 'kein', 'nichts', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'niemanden',
        #                  'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']
        # Rill + Wiegant et al. (2018)
        negation_list = ['nicht', 'kein', 'nichts', 'kaum', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'gegen',
                         'niemanden', 'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']
        # TODO: check further negation words in literature

    if get == 'candidates':

        # Loop over sentence parts and check whether a word is a noun/verb/adverb/adjective and append it as candidate
        for s in sent:
            c = []
            # loop over tokens in sentences, get tags and prepare
            for token in s:
                if verbose: print('token:', token.text, '->', token.tag_)
                if token.tag_.startswith(('NN', 'V', 'ADV', 'ADJ')) or token.text in negation_list:
                    if df_sepl['phrase'].str.contains(r'(?:\s|^){}(?:\s|$)'.format(token.text)).any():
                        c.append(token.text)
            candidates.append(c)

        if verbose: print('final candidates:', candidates)

    if get == 'negation':

        # loop over sentence parts and check whether a word is contained in negotion_list, if yes, append to candidates
        for s in sent:
            c = []
            # loop over tokens in sentence part
            for token in s:
                if verbose: print(token.text, token.tag_)
                if (token.text in negation_list):
                    # if (token.tag_.startswith(('PIAT', 'PIS', 'PTKNEG'))) or (token.text in negation_list):
                    c.append(token.text)
            candidates.append(c)
        if verbose: print('final negations:', candidates)

    return candidates


def ReadSePLSentiments(candidates, df_sepl=None, verbose=False):
    """
    reads in candidates (list in list), retrieves sentiment scores (sentiment_scores), returns them and the opinion
    relevant terms (tagged_phr), make sure df_sepl is loaded (run Load_SePL() before)

    Note: As Rill (2016) stated, this method does only work for simple sentences. To reduce complex candidates-lists,
    we only read in sentenceparts. However, it is possible that real life articles contain errors such as missing
    commas. This might result in incorrect identification of Sentiments which we cannot control for.

    :param candidates: list in list with POS tagged words
    :param df_sepl: load df_sepl via Load_SePL()
    :param verbose: display
    :return: [sentiment_scores], [tagged_phr]
    """

    final_sentiments, final_phrs, tagged_phr_list = [], [], []
    # loop over candidates and extract sentiment score according to Rill (2016): S.66-73, 110-124
    for c in candidates:
        c_sentiments, c_phrs = [], []
        # loop over each word in nested candidate list
        for word in c:
            stack = []
            index = c.index(word)
            if verbose: print('\n###### word:', word, 'index:', index, 'candidates:', c, '######')

            # check whether candidate is contained in SePL, if yes, get left and right neighbors
            if df_sepl['phrase_sorted'].str.contains(word).any():
                stack.append(word)
                if verbose: print(word, '|| stack without neighbours:', stack)
                for i in c[index + 1:]:
                    stack.append(i)
                    if verbose: print(word, '|| stack with right neighbours:', stack)

                # select slice of left neigbours and reverse it with second bracket
                for x in c[:index][::-1]:
                    stack.append(x)
                    if verbose: print(word, '|| stack with left neighbours:', stack)

                if verbose: print('final stack:', stack)

                # loop over stack and check whether word in SePL, if not, delete el from stack, if yes, extract sentiment,
                # delete el from stack and continue with remaining el in stack
                while len(stack) > 0:
                    phr = sorted(stack)
                    phr_string = ' '.join(phr)
                    if verbose: print('phr_string:', phr_string)

                    # if el of stack found in SePL, extract sentiment and save phrases
                    if (df_sepl['phrase_sorted'] == phr_string).any() and phr_string not in c_phrs and set(
                            tagged_phr_list).intersection(phr).__len__() == 0:
                        # extract sentiment, SePL sometimes contains non-unique entries, thus get the highest value
                        # if there are more than 1 sentiments
                        try:
                            sentiment_score = df_sepl.loc[
                                df_sepl['phrase_sorted'] == phr_string, 'sentiment'].item()
                        except ValueError:
                            sentiment_score = max(
                                df_sepl.loc[df_sepl['phrase_sorted'] == phr_string, 'sentiment'].to_list())
                        c_sentiments.append(sentiment_score)
                        if verbose: print('phrase found! sentiment is', sentiment_score)
                        # save phr
                        c_phrs.append(phr_string)
                        tagged_phr_list = phr_string.split()
                        break

                    # if el of stack not found, delete it and continue with next el in stack
                    else:
                        if verbose: print('deleting', stack[-1])
                        del stack[-1]

        # gather all extracted sentiments and phrases
        final_sentiments.append(c_sentiments)
        final_phrs.append(c_phrs)

    if verbose: print('final list with sentiments:', final_sentiments)
    if verbose: print('final list of phrs:', final_phrs)

    return final_sentiments, final_phrs


def ProcessSentimentScores(sepl_phrase, negation_candidates, sentimentscores, negation_list=None):
    """
    Process sentimentscores of sentence parts and return only one sentiment score per sentence/sentence part

        # Case I:
        # 1. any word contained in negation_list, sepl_phrase is already negated in SePL list
        # (2. sentimentscore is not empty)
        # -> do nothing

        # Case II:
        # 1. any word is NOT contained in negation_list, sepl_phrase is NOT negated in SePL list
        # (2. sentimentscore is not empty)
        # 3. negation_candidates is not empty
        # -> Invert sentimentscore

    :param sepl_phrase: GetSentiments(...)[1], here are all words which are in SePL
    :param negation_candidates: MakeCandidates(..., get='negation')
    :param sentimentscores: GetSentiments(...)[0]
    :return: 1 sentiment score
    """

    if negation_list is None:
        negation_list = ['nicht', 'kein', 'nichts', 'kaum', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'gegen',
                         'niemanden', 'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']

    # Loop over each sentence part and access each list (sepl_word/negation_candidates/sentimentscores) via index
    for i in range(0, len(sepl_phrase)):

        # Check whether sepl_word in sentence part is contained in negation_list, if yes, set flag to True
        # if sepl_phrase[i]:
        if sepl_phrase[i] and negation_candidates[i]:

            # write as str
            sepl_string = sepl_phrase[i][0]
            sepl_neg_string = negation_candidates[i][0]

            # set up flags
            seplphr, seplphrneg = False, False

            # check whether negation word in sepl_string, in sepl_neg_string
            for word in sepl_string.split():
                if word in negation_list: seplphr = True
            for word in sepl_neg_string.split():
                if word in negation_list: seplphrneg = True

            # Condition Case II: Invert sentiment
            if not seplphr and seplphrneg:
                sentimentscores[i][0] = -sentimentscores[i][0]
        else:
            continue

    # Flatten list
    flatsentimentscores = [element for sublist in sentimentscores for element in sublist]

    # Average sentiment score
    if flatsentimentscores:
        averagescore = sum(flatsentimentscores) / len(flatsentimentscores)
    else:
        averagescore = []

    return averagescore


def ProcessSePLphrases(sepl_phrase):
    """
    Process sepl_phrases of sentence parts and return only one list with the opinion relevant words per sentence,
    drop empty nested lists

    :param sepl_phrase: GetSentiments(...)[1], here are all words which are in SePL
    :return: 1 sepl_word list
    """

    # Loop over sentence parts and append only non-empty lists
    processed_sepl_phrases = ([])
    for phrase in sepl_phrase:
        if phrase:
            for p in phrase:
                processed_sepl_phrases.append(p)
    return processed_sepl_phrases


def GetSentimentScores(listOfSentenceparts, sentiment_list):
    """
    Run this function on each article (sentence- or paragraph-level) and get final sentiment scores.
    Note: Apply this function on the final long file only!

    Includes following function:

    1. Load_SePL() to load SePL
    2. MakeCandidates() to make candidates- and candidates_negation-lists
    3. ReadSePLSentiments() which reads in candidates- and candidates_negation-lists and retrieves sentiment scores
        from SePL
    4. ProcessSentimentScores() to process the retrieved sentiment scores and to return a unified score per sentence/sentence part

    :param listOfSentenceparts
        input must be processed by ProcessforSentiment() where listOfSentenceparts==1 sentence
        ['sentencepart', 'sentencepart', ...]
    :return: return 1 value per Article, return 1 list with sentiments of each sentence, 1 list w/ opinion relev. words
    """

    listOfSentiScores, listOfseplphrs = [], []

    for sentpart in listOfSentenceparts:
        """
        first step: identification of suitable candidates for opinionated phrases suitable candidates: 
        nouns, adjectives, adverbs and verbs
        """
        candidates = MakeCandidates(sentpart, sentiment_list, get='candidates')
        negation_candidates = MakeCandidates(sentpart, sentiment_list, get='negation')

        """
        second step: extraction of possible opinion-bearing phrases from a candidate starting from a candidate, 
        check all left and right neighbours to extract possible phrases. The search is terminated on a comma (POS tag $,), 
        a punctuation terminating a sentence (POS tag $.), a conjunction (POS-Tag KON) or an opinion-bearing word that is 
        already tagged. (Max distance determined by sentence lenght)
        If one of the adjacent words is included in the SePL, together with the previously extracted phrase, it is added to 
        the phrase.
        """

        raw_sentimentscores, raw_sepl_phrase = ReadSePLSentiments(candidates, sentiment_list)

        """
        third step: compare extracted phrases with SePL After all phrases have been extracted, they are compared with the 
        entries in the SePL. (everything lemmatized!) If no  match is found, the extracted Phrase is shortened by the last 
        added element and compared again with the SePL. This is repeated until a match is found.
        """

        # Make sure sepl_phrase, negation_candidates, sentimentscores are of same size
        assert len(raw_sepl_phrase) == len(raw_sentimentscores) == len(candidates) == len(negation_candidates)

        # export processed, flattened lists
        sentimentscores = ProcessSentimentScores(raw_sepl_phrase, negation_candidates, raw_sentimentscores)
        sepl_phrase = ProcessSePLphrases(raw_sepl_phrase)

        listOfSentiScores.append(sentimentscores)
        listOfseplphrs.append(sepl_phrase)

    # create flat, non-empty list with scores
    sentiscores = np.array([i for i in listOfSentiScores if i])

    # Retrieve statistics
    ss_mean, ss_median, ss_n, ss_sd = sentiscores.mean(), np.median(sentiscores), sentiscores.size, sentiscores.std()

    return {'mean': ss_mean, 'median': ss_median, 'n': ss_n, 'sd': ss_sd, 'sentiscores': listOfSentiScores,
            'phrs': listOfseplphrs}


def EstimateLDA(dataframecolumn, type=p['type'], no_below=50, no_above=0.9, num_topics=5, num_words=10,
                alpha='symmetric', eta=None,
                eval_every=10, iterations=50, random_state=None, verbose=True,
                distributed=False, chunksize=2000, passes=1, update_every=1, decay=0.5, offset=1.0,
                gamma_threshold=0.001, minimum_probability=0.01, ns_conf=None, minimum_phi_value=0.01,
                per_word_topics=False, callbacks=None, dtype=np.float32, save_model=False):
    """
    Estimates lda model based on the given training corpus (article, sentence or paragraph level)

    :param dataframecolumn: str, in the format: "[['str1', 'str2'], ['str3', ...], ...]",
           documents in string format to use as training corpus for the lda model
    :param no_below: int, cutoff words in the training corpus with frequency below a certain number
    :param no_above: float, cutoff words in the training corpus with frequency above a certain number
    :param num_topics: int, number of topics to be estimated in the lda model
    :param num_words: int, number of topics to be included in topics when printed and saved to disk (save_model=True)
    :param alpha: a priori belief about topic probabilities- specify 'auto' to learn asymmetric prior from data
    :param eta: a priori belief about word probabilities - specify 'auto' to learn asymmetric prior from data
    :param eval_every: log perplexity estimation frequency, higher values slows down training
    :param iterations: maximum number of iterations through the corpus when inferring the topic distribution
    :param random_state: set seed to generate random state - useful for reproducibility
    :param save_model: False, model and topics will be saved to {path_project}/lda/... if set to True
    :return: returns tuple of estimated lda model and text objects used for estimation
    """

    # Read in dataframe column and convert to list of lists
    templist = dataframecolumn.tolist()
    docsforlda = MakeListInLists(templist)
    # Create a dictionary representation of the documents and frequency filter
    dict_lda = Dictionary(docsforlda)
    dict_lda.filter_extremes(no_below=no_below, no_above=no_above)
    # Bag-of-words representation of the documents
    corpus_lda = [dict_lda.doc2bow(doc) for doc in docsforlda]

    if type == 'tfidf':
        tfidf = TfidfModel(corpus_lda)
        corpus_lda = tfidf[corpus_lda]

    # Make a index to word dictionary
    temp = dict_lda[0]  # This is only to "load" the dictionary
    id2word_lda = dict_lda.id2token
    # Display corpus for lda
    # pp.pprint(dict_lda.token2id)
    # pp.pprint(id2word_lda)
    if verbose: print('Number of unique tokens: {}'.format(len(dict_lda)))
    if verbose: print('Number of documents: {}'.format(len(corpus_lda)))

    lda_model = LdaModel(corpus=corpus_lda, id2word=id2word_lda, num_topics=num_topics,
                         alpha=alpha, eta=eta,
                         eval_every=eval_every, iterations=iterations, random_state=random_state,
                         distributed=distributed, chunksize=chunksize, passes=passes,
                         update_every=update_every,
                         decay=decay, offset=offset, gamma_threshold=gamma_threshold,
                         minimum_probability=minimum_probability, ns_conf=ns_conf,
                         minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics,
                         callbacks=callbacks, dtype=dtype
                         )

    # Print the topic keywords
    if verbose: pp.pprint(lda_model.print_topics(num_topics=num_topics, num_words=num_words))

    # Save model
    if save_model:
        # retrieve currmodel
        currmodel_ = '{}_{}_{}_{}_k{}'.format(type, p['POStag'],
                                              str(round(no_above, ndigits=2)),
                                              str(round(no_below, ndigits=3)),
                                              str(round(num_topics, ndigits=0)))
        print('\n\tsaving model to {}lda/{}/model_{}/...'.format(path_project, p['lda_level_fit'][0], currmodel_))
        os.makedirs(path_project + "lda/{}/model_{}".format(p['lda_level_fit'][0], currmodel_), exist_ok=True)
        temp_file = datapath(path_project + "lda/{}/model_{}/lda_model".format(p['lda_level_fit'][0], currmodel_))
        # save
        lda_model.save(temp_file)
        # To load pretrained model: lda = LdaModel.load(temp_file)
        # save topics
        topics_temp = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
        with open(path_project + "lda/{}/model_{}/topics.txt".format(p['lda_level_fit'][0], currmodel_), 'w') as f:
            f.write("\n".join(map(str, topics_temp)))

    return lda_model, docsforlda, dict_lda, corpus_lda


def GetTopics(doc, lda_model, dict_lda):
    """
    Uses a previously trained lda model to estimate the topic distribution of a document

    :param sent: 1 sentence from long df after preprocessing
    :param lda_model: estimated LDA model
    :return: Topic distribution for the document. list of tupels with topic id and its probability
    """
    # lemmatize doc
    doc = nlp2(doc)

    # loop over all tokens in sentence and lemmatize them, disregard punctuation
    lemmatized_doc = []
    for token in doc:
        if len(token.text) > 1:
            lemmatized_doc.append(token.lemma_)

    # Create BOW representation of doc to use as input for the LDA model
    doc_bow = dict_lda.doc2bow(lemmatized_doc)

    return lda_model.get_document_topics(doc_bow)


def GetDomTopic(doc, lda_model, dict_lda):
    """
    Uses a previously trained lda model to estimate the dominant topic of a document

    :param doc: 1 document as a string (e.g. articles_text, capitalized nouns, from 03a_PreprocessingArticles.py)
    :param lda_model: estimated LDA model
    :return: dominant topic id and its probability as a tupel
    """
    # initialize in spacy
    doc = nlp2(doc)

    # loop over all tokens in doc, disregard punctuation (length>1); note for lemmatiz. tokens need to be capitalized
    lemmatized_doc = []
    for token in doc:
        if len(token.text) > 1:
            lemmatized_doc.append(token.lemma_)

    # make back lower again
    lemmatized_doc_lower = [i.lower() for i in lemmatized_doc]

    # Create BOW representation of doc to use as input for the LDA model and retrieve dominant topic
    doc_bow = dict_lda.doc2bow(lemmatized_doc_lower)
    domdoc = max(lda_model.get_document_topics(doc_bow), key=lambda item: item[1])

    return domdoc


def MakeTopicsBOW(topic, dict_lda):
    """
       Help function for LDADistanceMetric.
       Creates BOW representation of topic distributions.

       :param topic: topic to be transformed in to BOW
       :param dict_lda: dictionary of LDA model
       :return: list of tuples, topic in BOW representation
       """
    # split on strings to get topics and the probabilities
    topic = topic[1].split('+')
    # list to store topic bows
    topic_bow = []
    for word in topic:
        # split topic probability and word
        prob, word = word.split('*')
        # get rid of spaces
        word = word.replace(" ", "").replace('"', '')
        # map topic words to dictionary id
        word_id = dict_lda.doc2bow([word])
        # append word_id and topic probability
        topic_bow.append((word_id[0][0], float(prob)))

    return topic_bow


def LDAHellinger(lda_model, dict_lda, num_topics=None, num_words=10):
    """
    This functions returns the average hellinger distance for all topic pairs in an LDA model.
    Includes following function:

    1. MakeTopicsBOW to create BOW representation of the LDA topic distributions

    :param lda_model: LDA model for which the distance metrics should be computed
    :param num_words: number of most relevant words in each topic to compare
    :return: float, returns average distance metric over all topic pairs with values between 0 and 1
    """

    # generate BOW representation of topic distributions
    if num_topics is None:
        num_topics = lda_model.num_topics

    # extract topic word presentations to list
    list = lda_model.show_topics(num_topics=num_topics, num_words=num_words)
    list_bow, sum = [], 0
    for topic in list:
        help = MakeTopicsBOW(topic, dict_lda)
        list_bow.append(help)

    # compute distance metric for each topic pair in list_bow
    for i in list_bow:
        for j in list_bow:
            dis = hellinger(i, j)
        sum = sum + dis
    print('computed average Hellinger distance')

    return sum / lda_model.num_topics


def LDAJaccard(lda_model, topn=10):
    """
    This functions returns the average jaccard distance for all topic pairs in an LDA model.
    Includes following function:

    1. MakeTopicsBOW to create BOW representation of the LDA topic distributions

    :param lda_model: LDA model for which the distance metrics should be computed
    :param topn: number of most relevant words in each topic to compare
    :return: float, returns average distance metric over all topic pairs with values between 0 and 1
    """

    topic_list, sum = [], 0
    for i in range(0, lda_model.num_topics):
        topic_list.append([tuple[0] for tuple in lda_model.show_topic(topicid=i, topn=topn)])

    # compute distance metric for each topic pair in list_bow
    for i in topic_list:
        for j in topic_list:
            print(i, j)
            dis = jaccard(i, j)
            print(dis)
            sum = sum + dis
    print('computed average Jaccard distance')

    return sum / lda_model.num_topics


def LDACoherence(lda_model, corpus, dictionary, texts):
    """
    Calculates coherence score for a lda model

    :param lda_model: previously trained lda model
    :param corpus: training corpus of the previously estimated lda model
    :param dictionary: dictionary corpus of the previously estimated lda model
    :param texts: documents used for the training corpus
    :return: coherence value c_v (see Röder et al. 2015)
    """

    # we use coherence measure c_v as suggested by Röder et al. 2015, because it has the highest correlation
    # with human interpretability

    # lda_model_cm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence="u_mass")
    lda_model_cm = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    print(lda_model_cm.get_coherence())

    return lda_model_cm.get_coherence()


def LDACalibration(dataframecolumn, topics_start=1, topics_limit=20, topics_step=1,
                   topn=10, num_words=25, metric='hellinger', type='standard',
                   no_below=50, no_above=0.9, alpha='symmetric', eta=None, eval_every=10, iterations=50,
                   random_state=None, verbose=False,
                   display_num_words=10, display_plot=True, save_plot=False, save_model=False):
    """
    Computes one of three evaluation metrics (jaccard, hellinger or coherence c_v)
    for a series of lda models using a topic range. The computed values for the metrics are displayed in a plot.

    :param topics_start: start of topic range
    :param topics_limit: end of topic range
    :param topics_step: steps in topic range
    :param dataframecolumn: documents in string format to use as training corpus for the lda model
    :param topn: number of most relevant words in each topic to compare (Jaccard)
    :param num_words: number of most relevant words in each topic to compare (Hellinger)
    :param metric: specify which metric to use (jaccard, hellinger, coherence or perplexity)
    :param no_below: cutoff words in the training corpus with frequency below a certain number
    :param no_above: cutoff words in the training corpus with frequency above a certain number
    :param alpha: a priori belief about topic probablities- specify 'auto' to learn asymmetric prior from data
    :param eta: a priori belief about word probabilities - specify 'auto' to learn asymmetric prior from data
    :param eval_every: log perplexity estimation frequency, higher values slows down training
    :param iterations: maximum number of iterations through the corpus when inferring the topic distribution
    :param random_state: set seed to generate random state - useful for reproducibility
    :param display_plot: set to false to not display a plot of the computed metric
    :param save_plot: set to false to not save plot
    :param display_num_topics: display number of topics as specified in EstimateLDA(), Note: only to display!
    :param save_model: save model as specified in EstimateLDA()
    :return: plots evaluation metric for lda models over the specified topic range
    """

    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]

    # check which type of metric specified
    if not isinstance(metric, list):
        print(metric, 'single metric')
        m = metric
        metric = []
        metric.append(m)

    # create metrics lists where values will be saved to
    jaccard_values, hellinger_values, coherence_values, perplexity_values = [], [], [], []

    model_list, metric_results = [], {}

    for num_topics in range(topics_start, topics_limit, topics_step):
        lda_results = EstimateLDA(dataframecolumn=dataframecolumn,
                                  no_below=no_below,
                                  no_above=no_above,
                                  num_topics=num_topics,
                                  alpha=alpha,
                                  eta=eta,
                                  eval_every=eval_every,
                                  iterations=iterations,
                                  random_state=random_state,
                                  type=type,

                                  # display, save options
                                  num_words=display_num_words,
                                  save_model=save_model
                                  )
        lda_model, docsforlda, dict_lda, corpus_lda = lda_results[0], lda_results[1], lda_results[2], lda_results[3]
        model_list.append(lda_model)

        #     if metric == 'jaccard':
        #         metric_values.append(LDAJaccard(topn=topn, lda_model=lda_model))
        #     if metric == 'hellinger':
        #         metric_values.append(LDAHellinger(num_words=num_words, lda_model=lda_model, num_topics=None, dict_lda=dict_lda))
        #     if metric == 'coherence':
        #         metric_values.append(LDACoherence(lda_model=lda_model, corpus=corpus_lda, dictionary=dict_lda, texts=docsforlda))
        #     if metric == 'perplexity':
        #         metric_values.append(lda_model.log_perplexity(corpus_lda))

        #     if verbose: print('num_topics: {}, metric: {}, metric values: {}'.format(num_topics, metric, metric_values))

        # if display_plot:
        #     fig = plt.figure()
        #     ax = plt.subplot(111)
        #     ax.plot(range(topics_start, topics_limit, topics_step), metric_values,
        #             label='metric: {}, type="{}", POStag="{}",\nno_below={}, no_above={}, alpha="{}", eta="{}"'.format(
        #                 metric, type, p['POStag'],
        #                 str(round(no_below, ndigits=2)), str(round(no_above, ndigits=3)),
        #                 alpha, eta, code))
        #     ax.legend()
        #     if save_plot:
        #         plt.savefig(path_project +
        #                     'calibration/{}/calibration_{}_{}/{}/'.format(p['lda_level_fit'][0], type, p['POStag'], metric)+
        #                     'Figure_nobelow{}_noabove{}_alpha{}_eta{}.png'.format(str(round(no_below, ndigits=2)),
        #                                                                           str(round(no_above, ndigits=3)),
        #                                                                           alpha, eta))
        #     plt.show(block=False)
        #     time.sleep(1.5)
        #     plt.close('all')

        for m in metric:
            if m == 'jaccard':
                jaccard_values.append(LDAJaccard(topn=topn, lda_model=lda_model))
            if m == 'hellinger':
                hellinger_values.append(
                    LDAHellinger(num_words=num_words, lda_model=lda_model, num_topics=None, dict_lda=dict_lda))
            if m == 'coherence':
                coherence_values.append(
                    LDACoherence(lda_model=lda_model, corpus=corpus_lda, dictionary=dict_lda, texts=docsforlda))
            if m == 'perplexity':
                perplexity_values.append(lda_model.log_perplexity(corpus_lda))

        if verbose: print('num_topics: {}'.format(num_topics))

    # dirty...
    if display_plot:

        for m in metric:

            # jaccard
            if m == 'jaccard':
                fig = plt.figure()
                ax = plt.subplot(111)
                ax.plot(range(topics_start, topics_limit, topics_step), jaccard_values,
                        label='metric: {}, type={}, POStag={},\nno_below={}, no_above={}, alpha={}, eta={}'.format(
                            'jaccard', type, p['POStag'], str(round(no_below, ndigits=2)),
                            str(round(no_above, ndigits=3)), alpha, eta))
                ax.legend()
                if save_plot:
                    plt.savefig(path_project +
                                'calibration/{}/calibration_{}_{}/{}/'.format(p['lda_level_fit'][0], type, p['POStag'],
                                                                              'jaccard') +
                                'Figure_nobelow{}_noabove{}_alpha{}_eta{}.png'.format(str(round(no_below, ndigits=2)),
                                                                                      str(round(no_above, ndigits=3)),
                                                                                      alpha, eta))
                plt.show(block=False)
                time.sleep(1)
                plt.close('all')

            # hellinger
            if m == 'hellinger':
                fig = plt.figure()
                ax = plt.subplot(111)
                ax.plot(range(topics_start, topics_limit, topics_step), hellinger_values,
                        label='metric: {}, type={}, POStag={},\nno_below={}, no_above={}, alpha={}, eta={}'.format(
                            'hellinger', type, p['POStag'], str(round(no_below, ndigits=2)),
                            str(round(no_above, ndigits=3)), alpha, eta))
                ax.legend()
                if save_plot:
                    plt.savefig(path_project +
                                'calibration/{}/calibration_{}_{}/{}/'.format(p['lda_level_fit'][0], type, p['POStag'],
                                                                              'hellinger') +
                                'Figure_nobelow{}_noabove{}_alpha{}_eta{}.png'.format(str(round(no_below, ndigits=2)),
                                                                                      str(round(no_above, ndigits=3)),
                                                                                      alpha, eta))
                plt.show(block=False)
                time.sleep(1)
                plt.close('all')

            # coherence
            if m == 'coherence':
                fig = plt.figure()
                ax = plt.subplot(111)
                ax.plot(range(topics_start, topics_limit, topics_step), coherence_values,
                        label='metric: {}, type={}, POStag={},\nno_below={}, no_above={}, alpha={}, eta={}'.format(
                            'coherence', type, p['POStag'], str(round(no_below, ndigits=2)),
                            str(round(no_above, ndigits=3)), alpha, eta))
                ax.legend()
                if save_plot:
                    plt.savefig(path_project +
                                'calibration/{}/calibration_{}_{}/{}/'.format(p['lda_level_fit'][0], type, p['POStag'],
                                                                              'coherence') +
                                'Figure_nobelow{}_noabove{}_alpha{}_eta{}.png'.format(str(round(no_below, ndigits=2)),
                                                                                      str(round(no_above, ndigits=3)),
                                                                                      alpha, eta))
                plt.show(block=False)
                time.sleep(1)
                plt.close('all')

            # perplexity
            if m == 'perplexity':
                fig = plt.figure()
                ax = plt.subplot(111)
                ax.plot(range(topics_start, topics_limit, topics_step), perplexity_values,
                        label='metric: {}, type={}, POStag={},\nno_below={}, no_above={}, alpha={}, eta={}'.format(
                            'perplexity', type, p['POStag'], str(round(no_below, ndigits=2)),
                            str(round(no_above, ndigits=3)), alpha, eta))
                ax.legend()
                if save_plot:
                    plt.savefig(path_project +
                                'calibration/{}/calibration_{}_{}/{}/'.format(p['lda_level_fit'][0], type, p['POStag'],
                                                                              'perplexity') +
                                'Figure_nobelow{}_noabove{}_alpha{}_eta{}.png'.format(str(round(no_below, ndigits=2)),
                                                                                      str(round(no_above, ndigits=3)),
                                                                                      alpha, eta))
                plt.show(block=False)
                time.sleep(1)
                plt.close('all')

    # save metric results
    for m in metric:
        if m == 'jaccard':
            metric_results['jaccard'] = jaccard_values
        if m == 'hellinger':
            metric_results['hellinger'] = hellinger_values
        if m == 'coherence':
            metric_results['coherence'] = coherence_values
        if m == 'perplexity':
            metric_results['perplexity'] = perplexity_values

    return model_list, metric_results


"""
################## Sentiment score functions to apply to long files ##################
"""


def ProcessforSentiment_l(sent):
    """
    Process sentences before running Sentiment Analysis, replace ;: KON by , and drop .!? and lemmatize
    :param listOfSents: list of sentences where sentences are str
    :return: listOfSentenceparts
        [['sentencepart1', 'sentencepart2', ...], [], [], ...]
        which are split by ,
    """
    listOfSents = [sent]  # new line
    temp_article, processed_article, final_article = [], [], []
    for sent in listOfSents:
        # First drop .?! and brackets
        temp_sent = sent.replace('.', '').replace('!', '').replace('?', '').replace('(', '').replace(')', '').replace(
            '[', '').replace(']', '')
        # Replace :; by ,
        temp_sent = temp_sent.replace(';', ',').replace(':', ',')
        # apply nlp2 to temp_sent
        temp_sent_nlp = nlp2(temp_sent)
        # process each token and 'translate' konjunction or ;: to ,
        temp_sent = []
        for token in temp_sent_nlp:
            if token.tag_ == 'KON':
                temp_sent.append(',')
            else:
                temp_sent.append(token.text)

        # put all tokens to a string (but split later by normalized ,)
        sent_string = ' '.join(temp_sent)

        # prepare for lemmatization
        sent_string = nlp2(sent_string)

        # Second, loop over all tokens in sentence and lemmatize them
        sent_tokens = []
        for token in sent_string:
            sent_tokens.append(token.lemma_)
        processed_article.append(sent_tokens)

    # Put together tokenized, lemmatized elements of lists to a string
    processed_article = [' '.join(i) for i in processed_article]
    # Split by normalized commas
    for sent in processed_article:
        final_article.append(sent.split(','))

    # Flatten list
    final_article = FlattenList(final_article)

    # strip strings
    final_article = [x.strip() for x in final_article]

    # drop empty elements
    final_article = [x for x in final_article if x != '']
    # final_article = [x.strip() for x in final_article if x.strip()]

    #  return a string with lemmatized words and united sentence splits to ,
    return final_article


def GetSentimentScores_l(sent, sentiment_list, verbose=False):
    """
    Run this function on each article (sentence- or paragraph-level) and get final sentiment scores.
    Note: Apply this function on the final long file only!

    Includes following function:

    1. Load_SePL() to load SePL
    2. MakeCandidates() to make candidates- and candidates_negation-lists
    3. ReadSePLSentiments() which reads in candidates- and candidates_negation-lists and retrieves sentiment scores
        from SePL
    4. ProcessSentimentScores() to process the retrieved sentiment scores and to return a unified score per sentence/sentence part

    :param listOfSentenceparts
        input must be processed by ProcessforSentiment() where listOfSentenceparts equals 1 sentence
        ['sentencepart', 'sentencepart', ...]
    :return: return 1 value per Article, return 1 list with sentiments of each sentence, 1 list w/ opinion relev. words
    """
    if verbose: print('### sent:', sent)
    listOfSentenceparts = ProcessforSentiment_l(sent)
    listOfSentiScores, listOfphrs = [], []

    for sentpart in listOfSentenceparts:
        """
        first step: identification of suitable candidates for opinionated phrases suitable candidates: 
        nouns, adjectives, adverbs and verbs
        """
        # if verbose: print('\tsentpart:', sentpart, end='\r')
        candidates = MakeCandidates(sentpart, sentiment_list, get='candidates')
        negation_candidates = MakeCandidates(sentpart, sentiment_list, get='negation')
        # if verbose: print('\tcandidates:', candidates, end='\r')
        # if verbose: print('\tnegation_candidates:', negation_candidates)
        """
        second step: extraction of possible opinion-bearing phrases from a candidate starting from a candidate, 
        check all left and right neighbours to extract possible phrases. The search is terminated on a comma (POS tag $,), 
        a punctuation terminating a sentence (POS tag $.), a conjunction (POS-Tag KON) or an opinion-bearing word that is 
        already tagged. (Max distance determined by sentence lenght)
        If one of the adjacent words is included in the SePL, together with the previously extracted phrase, it is added to 
        the phrase.
        """

        raw_sentimentscores, raw_phrs = ReadSePLSentiments(candidates, sentiment_list)
        # if verbose: print('\traw_sentimentscores:', raw_sentimentscores, 'raw_sepl_phrase:', raw_sepl_phrase)

        """
        third step: compare extracted phrases with SePL After all phrases have been extracted, they are compared with the 
        entries in the SePL. (everything lemmatized!) If no  match is found, the extracted Phrase is shortened by the last 
        added element and compared again with the SePL. This is repeated until a match is found.
        """

        # Make sure sepl_phrase, negation_candidates, sentimentscores are of same size
        assert len(raw_phrs) == len(raw_sentimentscores) == len(candidates) == len(negation_candidates)

        # export processed, flattened lists
        sentimentscores = ProcessSentimentScores(raw_phrs, negation_candidates, raw_sentimentscores)
        # if verbose: print('\tsentimentscores:', sentimentscores, end='\r')
        final_phrs = ProcessSePLphrases(raw_phrs)
        # if verbose: print('\tsepl_phrase:', sepl_phrase)

        listOfSentiScores.append(sentimentscores)
        listOfphrs.append(final_phrs)

    # create flat, non-empty list with scores
    sentiscores = np.array([i for i in listOfSentiScores if i])

    # Retrieve statistics
    ss_mean, ss_median, ss_n, ss_sd = sentiscores.mean(), np.median(sentiscores), sentiscores.size, sentiscores.std()
    if verbose: print('\tstats:', ss_mean, ss_median, ss_n, ss_sd, end='\n\n')

    return {'mean': ss_mean, 'median': ss_median, 'n': ss_n, 'sd': ss_sd, 'sentiscores': listOfSentiScores,
            'phrs': listOfphrs}


#######################################################################################################################
######################################################################################################################
# Functions for SentiWS/simple word lists

def Load_SentiWS(type='default'):
    """
    Reads in SentiWS, prepares phrases and sorts them; this is required be be run before MakeCandidatesWS() and
    GetSentiments()
    Note: SentiWS.csv -> SentiWS_final.csv, so both pos/neg columns in SentiWS.csv are appended together,
          so SentiWS_final.csv containes unedited sentiment scores / phrs

    :param type: default 'default' (calls SentiWS_final.csv), 'modified' (calls SentiWS_final_v1.x_negated_modified.csv)
    :return: Pandas dataframe with sentiws-phrs and sentiments
    """
    if type == 'modified':
        # Read in modified SentiWS
        df_SentiWS = pandas.read_csv(path_data + 'Sentiment/SentiWS/SentiWS_final_v1.x_negated_modified.csv', sep=';')
    else:
        # Read in default SentiWS
        df_SentiWS = pandas.read_csv(path_data + 'Sentiment/SentiWS/SentiWS_final.csv', sep=';')

    # convert all words to lower case
    df_SentiWS['word'] = [i.lower() for i in df_SentiWS['word']]

    print('SentiWS ({}) file loaded'.format(type))

    return df_SentiWS


def MakeCandidatesWS(sent, df_SentiWS=None, get='candidates', verbose=False, negation_list=None):
    sent = sent.split(',')
    sent = [nlp2(s) for s in sent]
    candidates = []

    if negation_list is None:
        # Rill (2016)
        # negation_list = ['nicht', 'kein', 'nichts', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'niemanden',
        #                  'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']
        # Rill + Wiegant et al. (2018)
        negation_list = ['nicht', 'kein', 'nichts', 'kaum', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'gegen',
                         'niemanden', 'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']
        # TODO: check further negation words in literature

    if get == 'candidates':

        # Loop over sentence parts and check whether a word is a noun/verb/adverb/adjective and append it as candidate
        for s in sent:
            c = []
            # loop over tokens in sentences, get tags and prepare
            for token in s:
                # if verbose: print('token:', token.text, '->', token.tag_)
                if token.tag_.startswith(('NN', 'V', 'ADV', 'ADJ')) or token.text in negation_list:
                    if df_SentiWS['word'].str.contains(r'(?:\s|^){}(?:\s|$)'.format(token.text)).any():
                        c.append(token.text)
            candidates.append(c)

        if verbose: print('final candidates:', candidates)

    if get == 'negation':

        # loop over sentence parts and check whether a word is contained in negation_list, if yes, append to candidates
        for s in sent:
            c = []
            # loop over tokens in sentence part
            for token in s:
                # if verbose: print(token.text, token.tag_)
                if (token.text in negation_list):
                    # if (token.tag_.startswith(('PIAT', 'PIS', 'PTKNEG'))) or (token.text in negation_list):
                    c.append(token.text)
            candidates.append(c)
        if verbose: print('final negations:', candidates)

    return candidates


def ReadSentiWSSentiments(candidates, df_SentiWS=None, verbose=False):
    """
    reads in candidates (list in list), retrieves sentiment scores (sentiment_scores), returns them and the opinion
    relevant terms (tagged_phr), make sure df_SentiWS is loaded (run Load_SentiWS() before)

    :param candidates: list in list with POS tagged words
    :param df_SentiWS: load df_SentiWS via Load_SentiWS()
    :param verbose: display
    :return: [sentiment_scores], [tagged_phr]
    """

    final_sentiments, final_phrs = [], []
    # loop over candidates and extract sentiment score according to Rill (2016): S.66-73, 110-124
    for c in candidates:
        c_sentiments, c_phrs = [], []
        # loop over each word in nested candidate list
        for word in c:
            # check whether candidate is contained in SentiWS, if yes, get left and right neighbors
            # if (df_SentiWS['word'] == word).any():
            if df_SentiWS['word'].str.contains(word).any():
                # extract sentiment - if SentiWS contains non-unique entries, get the highest value
                # if there are more than 1 sentiments
                try:
                    sentiment_score = df_SentiWS.loc[df_SentiWS['word'] == word, 'sentiment'].item()
                    if verbose: print(sentiment_score, word)
                except ValueError:
                    sentiment_score = max(df_SentiWS.loc[df_SentiWS['word'] == word, 'sentiment'].to_list())
                c_sentiments.append(sentiment_score)
                if verbose: print('phrase found! sentiment is', sentiment_score)
                # save phr
                c_phrs.append(word)

        # gather all extracted sentiments and phrases
        final_sentiments.append(c_sentiments)
        final_phrs.append(c_phrs)

    if verbose: print('final list with sentiments:', final_sentiments)
    if verbose: print('final list of phrs:', final_phrs)

    return final_sentiments, final_phrs


def ProcessSentimentScoresWS(WS_phrase, negation_candidates, sentimentscores, negation_list=None):
    """
    Process sentimentscores of sentence parts and return only one sentiment score per sentence/sentence part
    negation: senti score of first opinion word in sentence part is negated if negation present
              in the same sentence part

    :param sepl_phrase: GetSentiments(...)[1], here are all words which are in SePL
    :param negation_candidates: MakeCandidates(..., get='negation')
    :param sentimentscores: GetSentiments(...)[0]
    :return: 1 sentiment score

    Todo: negate highest sent in sentence part or just leave it as it is?

    """

    if negation_list is None:
        negation_list = ['nicht', 'kein', 'nichts', 'kaum', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'gegen',
                         'niemanden', 'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']

    # Loop over each sentence part and access each list (SentiWS_word/negation_candidates/sentimentscores) via index
    for i in range(0, len(WS_phrase)):

        # Check whether SentiWS_word in sentence part is contained in negation_list, if yes, set flag to True
        # if WS_phrase[i]:
        if WS_phrase[i] and negation_candidates[i]:

            # # write as str
            # WS_string = WS_phrase[i][0]
            # WS_neg_string = negation_candidates[i][0]
            #
            # # set up flags
            # WSphr, WSphrneg = False, False
            #
            # # check whether negation word in SentiWS_string, in SentiWS_neg_string
            # for word in WS_string.split():
            #     if word in negation_list: WSphr = True
            # for word in WS_neg_string.split():
            #     if word in negation_list: WSphrneg = True
            #
            # # Condition Case II: Invert sentiment
            # if not WSphr and WSphrneg:
            sentimentscores[i][0] = -sentimentscores[i][0]
        else:
            continue

    # Flatten list
    flatsentimentscores = [element for sublist in sentimentscores for element in sublist]

    # Average sentiment score
    if flatsentimentscores:
        averagescore = sum(flatsentimentscores) / len(flatsentimentscores)
    else:
        averagescore = []

    return averagescore


def ProcessSentiWSphrases(WS_phrase):
    """
    Process WS_phrases of sentence parts and return only one list with the opinion relevant words per sentence,
    drop empty nested lists

    :param WS_phrase: GetSentiments(...)[1], here are all words which are in SentiWS
    :return: 1 WS_word list
    """

    # Loop over sentence parts and append only non-empty lists
    processed_WS_phrases = ([])
    for phrase in WS_phrase:
        if phrase:
            for p in phrase:
                processed_WS_phrases.append(p)
    return processed_WS_phrases


def GetSentimentScoresWS(listOfSentenceparts, df_SentiWS):
    """
    Run this function on each article (sentence- or paragraph-level) and get final sentiment scores.
    Note: Apply this function on the final long file only!

    Includes following function:

    1. Load_SentiWS() to load SentiWS
    2. MakeCandidatesWS() to make candidates- and candidates_negation-lists
    3. ReadWSSentiments() which reads in candidates- and candidates_negation-lists and retrieves sentiment scores
        from SentiWS
    4. ProcessSentimentScoresWS() to process the retrieved sentiment scores and to return a unified score per sentence/sentence part

    :param listOfSentenceparts
        input must be processed by ProcessforSentiment() where listOfSentenceparts==1 sentence
        ['sentencepart', 'sentencepart', ...]
    :return: return 1 value per Article, return 1 list with sentiments of each sentence, 1 list w/ opinion relev. words
    """

    listOfSentiScores, listOfWSphrs = [], []

    for sentpart in listOfSentenceparts:
        """
        first step: identification of suitable candidates for opinionated phrases suitable candidates:
        nouns, adjectives, adverbs and verbs
        """
        candidates = MakeCandidates(sentpart, df_SentiWS, get='candidates')
        negation_candidates = MakeCandidates(sentpart, df_SentiWS, get='negation')

        """
        second step: extraction of possible opinion-bearing phrases from a candidate starting from a candidate,
        check all left and right neighbours to extract possible phrases. The search is terminated on a comma (POS tag $,),
        a punctuation terminating a sentence (POS tag $.), a conjunction (POS-Tag KON) or an opinion-bearing word that is
        already tagged. (Max distance determined by sentence lenght)
        If one of the adjacent words is included in the SentiWS, together with the previously extracted phrase, it is added to
        the phrase.
        """
        raw_sentimentscores, raw_WS_phrase = ReadSentiWSSentiments(candidates, df_SentiWS)

        """
        third step: compare extracted phrases with SentiWS After all phrases have been extracted, they are compared with the
        entries in the SentiWS (everything lemmatized!) If no  match is found, the extracted Phrase is shortened by the last
        added element and compared again with the SentiWS This is repeated until a match is found.
        """

        # Make sure SentiWS_phrase, negation_candidates, sentimentscores are of same size
        assert len(raw_WS_phrase) == len(raw_sentimentscores) == len(candidates) == len(negation_candidates)

        # export processed, flattened lists
        sentimentscores = ProcessSentimentScoresWS(raw_WS_phrase, negation_candidates, raw_sentimentscores)
        WS_phrase = ProcessSentiWSphrases(raw_WS_phrase)

        listOfSentiScores.append(sentimentscores)
        listOfWSphrs.append(WS_phrase)

    # create flat, non-empty list with scores
    sentiscores = np.array([i for i in listOfSentiScores if i])

    # Retrieve statistics
    ss_mean, ss_median, ss_n, ss_sd = sentiscores.mean(), np.median(sentiscores), sentiscores.size, sentiscores.std()

    return {'mean': ss_mean, 'median': ss_median, 'n': ss_n, 'sd': ss_sd}, listOfSentiScores, listOfWSphrs


def GetSentimentScoresWS_l(sent, df_SentiWS, verbose=False):
    """
    Run this function on each article (sentence- or paragraph-level) and get final sentiment scores.
    Note: Apply this function on the final long file only!

    Includes following function:

    1. Load_SentiWS() to load SentiWS
    2. MakeCandidates() to make candidates- and candidates_negation-lists
    3. ReadSentiWSSentiments() which reads in candidates- and candidates_negation-lists and retrieves sentiment scores
        from SentiWS
    4. ProcessSentimentScoresWS() to process the retrieved sentiment scores and to return a unified score per sentence/sentencepart

    :param listOfSentenceparts
        input must be processed by ProcessforSentiment() where listOfSentenceparts equals 1 sentence
        ['sentencepart', 'sentencepart', ...]
    :return: return 1 value per Article, return 1 list with sentiments of each sentence, 1 list w/ opinion relev. words
    """
    if verbose: print('### sent:', sent)
    listOfSentenceparts = ProcessforSentiment_l(sent)
    listOfSentiScores, listOfWSphrs = [], []

    for sentpart in listOfSentenceparts:
        """
        first step: identification of suitable candidates for opinionated phrases suitable candidates:
        nouns, adjectives, adverbs and verbs
        """
        # if verbose: print('\tsentpart:', sentpart, end='\r')
        candidates = MakeCandidatesWS(sentpart, df_SentiWS, get='candidates')
        negation_candidates = MakeCandidatesWS(sentpart, df_SentiWS, get='negation')
        # if verbose: print('\tcandidates:', candidates, end='\r')
        # if verbose: print('\tnegation_candidates:', negation_candidates)
        """
        second step: extraction of possible opinion-bearing phrases from a candidate starting from a candidate,
        check all left and right neighbours to extract possible phrases. The search is terminated on a comma (POS tag $,),
        a punctuation terminating a sentence (POS tag $.), a conjunction (POS-Tag KON) or an opinion-bearing word that is
        already tagged. (Max distance determined by sentence lenght)
        If one of the adjacent words is included in the SentiWS, together with the previously extracted phrase, it is added to
        the phrase.
        """
        raw_sentimentscores, raw_WS_phrase = ReadSentiWSSentiments(candidates, df_SentiWS)

        """
        third step: compare extracted phrases with SentiWS After all phrases have been extracted, they are compared with the
        entries in the SentiWS. (everything lemmatized!) If no  match is found, the extracted Phrase is shortened by the last
        added element and compared again with the SentiW. This is repeated until a match is found.
        """

        # Make sure WS_phrase, negation_candidates, sentimentscores are of same size
        assert len(raw_WS_phrase) == len(raw_sentimentscores) == len(candidates) == len(negation_candidates)

        # export processed, flattened lists
        sentimentscores = ProcessSentimentScoresWS(raw_WS_phrase, negation_candidates, raw_sentimentscores)
        # if verbose: print('\tsentimentscores:', sentimentscores, end='\r')
        WS_phrase = ProcessSentiWSphrases(raw_WS_phrase)

        listOfSentiScores.append(sentimentscores)
        listOfWSphrs.append(WS_phrase)

    # create flat, non-empty list with scores
    sentiscores = np.array([i for i in listOfSentiScores if i])

    # Retrieve statistics
    ss_mean, ss_median, ss_n, ss_sd = sentiscores.mean(), np.median(sentiscores), sentiscores.size, sentiscores.std()
    if verbose: print('\tstats:', ss_mean, ss_median, ss_n, ss_sd, end='\n\n')

    return {'mean': ss_mean, 'median': ss_median, 'n': ss_n, 'sd': ss_sd, 'sentiscores': listOfSentiScores,
            'phrs': listOfWSphrs}

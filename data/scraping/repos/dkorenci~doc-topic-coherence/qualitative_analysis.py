from pytopia.context.ContextResolver import resolve
from doc_topic_coh.resources import pytopia_context

from doc_topic_coh.coherence.coherence_builder import CoherenceFunctionBuilder
from pytopia.measure.topic_distance import cosine
from doc_topic_coh.dataset.topic_splits import uspolTopicsLabeled
from doc_topic_coh.dataset.topic_splits import devTestSplit

from pytopia.context.ContextResolver import resolve

cacheFolder = '/datafast/doc_topic_coh/experiments/function_cache/'

def topicLabel(t):
    '''
    :param t: (modelId, topicId)
    '''
    return '%s.%s'%t

# requires corpus_topic_index_builder
def printDocumentTitles(topic, topDocs=10, corpus='us_politics'):
    '''
    :param topic: (modelId, topicId)
    :param topDocs:
    :return:
    '''
    mid, tid = topic
    ctiBuilder = resolve('corpus_topic_index_builder')
    cti = ctiBuilder(corpus=corpus, model=mid)
    wtexts = cti.topicTexts(tid, top=topDocs)
    txtIds = [ id_ for id_, _ in wtexts ]
    corpus = resolve(corpus)
    idTexts = corpus.getTexts(txtIds)
    for txto in idTexts:
        print txto.title

def listTopics(coh, topics, topics2print=10, top=True):
    '''
    List coherence scores and labels of top/bottom topics.
    :param coh: coherence measure
    :param topics: list of labeled topics
    :param topics2print: top words for topic label
    :param top: display top or bottom topics
    :return:
    '''
    res = []
    for t, tl in topics:
        res.append((coh(t), t))
    res.sort(key=lambda p:p[0], reverse=top)
    for i in range(topics2print):
        c = res[i][0]
        topic = res[i][1]
        mi, ti = topic
        model = resolve(mi)
        print '%15s: %s , %.4f' % (topicLabel(topic), model.topic2string(ti, 10), c)
        printDocumentTitles(topic)
        print

def contrastCoherences(coh1, coh2, topics, tableParse, ltopics, coh1Top=True, coh2Top=False,
                       sort=None, per1=0.9, per2=0.1, topWords=10):
    '''
    Display topics with good rank by one coherence measure and
     bad ranked by another measure.
    :param per1, per2: percentiles that define what is good and bad rank -
        take per1 percentile by coh1 (or above) and bottom per2 percentile by coh2 (or below)
    :param topics: list of labeled topics
    :param topWords: top words for topic label
    :return:
    '''
    from numpy import percentile
    res1 = [coh1(t) for t, tl in topics]
    res2 = [coh2(t) for t, tl in topics]
    perc1 = percentile(res1, per1*100.0)
    perc2 = percentile(res2, per2*100.0)
    print 'coh_top', coh1.id
    print 'coh_bot', coh2.id
    selected = []
    selector = lambda score, perc, above: score >= perc if above else score <= perc
    selector1 = lambda score: selector(score, perc1, coh1Top)
    selector2 = lambda score: selector(score, perc2, coh2Top)
    for i, t in enumerate(topics):
        if selector1(res1[i]) and selector2(res2[i]):
            topic = t[0]
            selected.append(topic)
    if sort:
        if sort == coh1:
            selected = sorted(selected, key=lambda t: coh1(t), reverse=True)
        else:
            selected = sorted(selected, key=lambda t: coh2(t), reverse=True)
    topic2label = {t: l for t, l in ltopics}
    for topic in selected:
        mi, ti = topic
        model = resolve(mi)
        label = topic2label[topic]
        semtopics = u';'.join(th for th in tableParse.getTopic(topicLabel(topic)).themes)
        print '%15s: %s , %s , [%s]' % (topicLabel(topic), model.topic2string(ti, topWords),
                                        label, semtopics)

def scorer(params, cache=None):
    if cache: params['cache'] = cache
    return CoherenceFunctionBuilder(**params)()

def documentVsWordCoherence():
    bestGraph1 = { 'distance': cosine, 'weighted': False, 'center': 'mean',
                    'algorithm': 'communicability', 'vectors': 'tf-idf',
                    'threshold': 50, 'weightFilter': [0, 0.92056], 'type': 'graph' }
    cp = { 'type':'c_p', 'standard': False, 'index': 'wiki_docs', 'windowSize': 70}
    cohDoc = scorer(bestGraph1, cacheFolder)
    cohWord = scorer(cp, cacheFolder)
    dev, test = devTestSplit()
    topics = test
    from doc_topic_coh.dataset.topic_labeling import tableParse
    tparse, ltopics = tableParse(), uspolTopicsLabeled()
    print '********************** high doc low word **********************'
    contrastCoherences(cohDoc, cohWord, topics, tparse, ltopics,
                       per1=0.7, per2=0.3, coh1Top=True, coh2Top=False, sort=cohWord)
    print '********************** high doc high word **********************'
    contrastCoherences(cohWord, cohDoc, topics, tparse, ltopics,
                       per1=0.7, per2=0.3, coh1Top=True, coh2Top=True, sort=cohWord)
    print '********************** low doc high word **********************'
    contrastCoherences(cohWord, cohDoc, topics, tparse, ltopics,
                       per1=0.7, per2=0.3, coh1Top=True, coh2Top=False, sort=cohWord)
    print '********************** low doc low word **********************'
    contrastCoherences(cohWord, cohDoc, topics, tparse, ltopics,
                       per1=0.3, per2=0.3, coh1Top=False, coh2Top=False, sort=cohWord)

from pytopia.nlp.text2tokens.gtar.text2tokens import RsssuckerTxt2Tokens
def destemWords(words, top=True, corpusId='us_politics', text2tokens=RsssuckerTxt2Tokens()):
    itb = resolve('inverse_tokenizer_builder')
    itok = itb(corpusId, text2tokens, True)
    if top:
        print ' '.join(itok.allWords(w)[0] for w in words.split())
    else:
        for w in words.split():
            print w, itok.allWords(w)

if __name__ == '__main__':
    documentVsWordCoherence() # Section 5.3
from doc_topic_coh.coherence.experiment import TopicScoringExperiment as TSE
from doc_topic_coh.coherence.coherence_builder import CoherenceFunctionBuilder
from doc_topic_coh.settings import experiments_folder
from doc_topic_coh.coherence.tools import IdList


def subsampleList(l, size, rndseed = 123456):
    '''
    Create an id-able list as a subsample of an id-able list
    :param l: list like, id-able
    '''
    from random import sample, seed
    seed(rndseed)
    ss = IdList(sample(l, size))
    ss.id = l.id + '_subsample[size=%d,seed=%d]' % (size, rndseed)
    return ss

def assignVectors(params, vectors, modifyId=True):
    '''
    Assigning 'vectors' parameter to each member of params,
     for each possible value from the set of values,
     set of values is either 'corpus' or 'world' vectors.
    '''
    if vectors == 'world': vec = ['word2vec', 'glove', 'word2vec-avg', 'glove-avg']
    elif vectors == 'corpus': vec = ['tf-idf', 'probability']
    else: raise Exception('invalid vectors')
    newp = IdList()
    for p in params:
        for vp in vec:
            np = p.copy()
            np['vectors'] = vp
            newp.append(np)
    if modifyId: newp.id = params.id + '_%s_vectors' % vectors
    else: newp.id = params.id
    return newp


def experiment(params, topics, action='run', vectors=None,
               evalTopics=None, plotEval=False, evalPerc=None,
               evalThresh=None, th2per=None, posClass=['theme', 'theme_noise'],
               scoreInd=None, expFolder=experiments_folder):
    if vectors: params = assignVectors(params, vectors)
    print params.id
    print 'num params', len(params)
    tse = TSE(paramSet=params, scorerBuilder=CoherenceFunctionBuilder,
              ltopics=topics, posClass=posClass,
              folder=expFolder, cache=True)
    if action == 'run': tse.run()
    elif action == 'print': tse.printResults()
    elif action == 'signif': tse.significance(scoreInd=scoreInd)
    elif action == 'eval':
        tse.evalOnTopics(evalTopics, plot=plotEval, percentile=evalPerc, saveDev=False)
    elif action == 'printTop':
        tse.evalOnTopicsPrintTop(evalTopics, thresh=evalThresh, percentile=evalPerc,
                                 th2per=th2per)
    else: print 'specified action not defined'
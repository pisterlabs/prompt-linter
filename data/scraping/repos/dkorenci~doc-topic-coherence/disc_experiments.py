from doc_topic_coh.topic_discovery.dataset import *

from doc_topic_coh.coherence.measure_evaluation.evaluations import palmettoCp, docCohBaseline
from doc_topic_coh.coherence.coherence_builder import CoherenceFunctionBuilder
from doc_topic_coh.topic_discovery.theme_count import \
    AggregateThemeCount, TopicDistEquality, sortByCoherence, plotThemeCounts
from pytopia.measure.topic_distance import cosine


def plotTopicDiscovery(modelParams, topics, numRandom=5, rseed=102, save = None, maxx=None,
                       cache='/datafast/doc_topic_coherence/experiments/iter5_coherence/function_cache'):
    from copy import copy
    topics = copy(topics)
    if not isinstance(modelParams, list): modelParams = [modelParams]
    thCount = AggregateThemeCount(TopicDistEquality(cosine, 0.5), verbose=False)
    # evaluate counts for models
    cohCounts = []
    for param in modelParams:
        coh = CoherenceFunctionBuilder(cache=cache, **param)()
        topics = sortByCoherence(topics, coh)
        cohCounts.append(thCount(topics))
    # evaluate random counts
    from random import seed, shuffle
    seed(rseed)
    rndCounts = []
    for i in range(numRandom):
        shuffle(topics)
        rndcnt = thCount(topics)
        rndCounts.append(rndcnt)
    plotThemeCounts(cohCounts, rndCounts, maxx=maxx)

topics_nomix = allTopicsNoMix()
topics_mixempty = allTopicsMixedEmpty()

def plotAllModels(topics, numRandom=5):
    topGraphDocCohsTest = [
        {'distance': 'cosine', 'weighted': False, 'center': 'mean', 'algorithm': 'communicability',
         'vectors': 'tf-idf', 'threshold': 50, 'weightFilter': [0, 0.92056], 'type': 'graph'},
        {'distance': 'l2', 'weighted': False, 'center': 'mean', 'algorithm': 'communicability', 'vectors': 'tf-idf',
         'threshold': 50, 'weightFilter': [0, 1.37364], 'type': 'graph'}
    ]
    params = topGraphDocCohsTest
    params.extend([docCohBaseline, palmettoCp])
    plotTopicDiscovery(params, topics, numRandom=numRandom)

if __name__ == '__main__':
    plotAllModels(topics_mixempty, numRandom=20) # Section 6.2 Figure 2





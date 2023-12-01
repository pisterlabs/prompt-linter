# Library imports
from gensim.models import coherencemodel, ldamodel, wrappers
import os
import sys
# Add James to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# Project imports
from api.jamesClasses import jamesResults
from api.jamesConfig import cfg

def buildTopicModel(corpus, topicNum):
    '''
    This method is used to build a gensim topic model with the given number of
    topics for a given corpus. 

    Parameters
    ----------
            corpus: jamesCorpus
                    the corpus to be modeled, as a jamesCorpus
                    object (imported from jamesClasses)

            topicNum: int
                    the number of topics to generate

    Output
    ------
            gensim.models.ldamodel
                    the topic model generated from the input corpus
    '''
    # Add the path to mallet, imported from jamesConfig, to the environment
    os.environ['MALLET_HOME'] = cfg['path']['malletpath']
    # Build the topic model for the given number of topics using mallet, which
    #   is built locally, and a mallet wrapper imported from gensim.models 
    malletModel = wrappers.LdaMallet(cfg['path']['malletfile'], corpus=corpus.getBoW(), num_topics=topicNum, id2word=corpus.dic,
                                       random_seed=cfg['malletsettings']['random_seed'])
    # Convert the mallet model to an ldamodel
    ldaModel = wrappers.ldamallet.malletmodel2ldamodel(malletModel,
                                                       gamma_threshold=cfg['malletsettings']['gamma_threshold'],
                                                       iterations=cfg['malletsettings']['iterations'])
    # Return the topic model
    return ldaModel

def buildCoherenceModel(topicModel, corpus):
    '''
    This method is used to construct a coherencemodel (imported from gensim.models)
    from a generated topic model and the corpus.

    Parameters
    ----------
            topicModel: gensim.models.ldamodel
                    the topic model used to build the coherence model

            corpus: jamesCorpus
                    the corpus used to generate the topic model

    Output
    ------
            gensim.models.coherencemodel
                    a coherence model (imported from gensim.models) for the input
                    topic model
    '''
    return coherencemodel.CoherenceModel(model=topicModel, texts=corpus.getLemmatized(),
                                                   dictionary=corpus.dic, corpus=corpus.getBoW(),
                                                   coherence=cfg['coherencetype'])

def getResults(topicModel, coherenceModel, corpus):
    '''
    This method is used to construct a jamesResults object (imported from jamesClasses)
    containing the topic results of a given topic model, coherence model, and corpus

    Parameters
    ----------
            topicModel: gensim.models.ldamodel
                    the topic model whose results are being returned

            coherenceModel: gensim.models.coherencemodel
                    the coherence model for the given topic model

            corpus: jamesCorpus
                    the corpus used to generate the input topic model as a
                    jamesCorpus object (imported from jamesClasses)

    Output
    ------
            jamesResults
                    a jamesResults object (imported from jamesClasses) containing
                    the topic results
    '''
    return jamesResults([topic[0] for topic in topicModel.top_topics(corpus.getBoW(),topn=cfg['topicwords'])],
                        float(coherenceModel.get_coherence()),
                        [float(coherence) for coherence in coherenceModel.get_coherence_per_topic()])

def getTopics(bow, topicModel):
    '''
    This method is used to find the topic distribution of a given document or sentence
    It is used by jamesMain to find the topic distribution of documents for the result set,
    and to find the topic distribution of each sentence for sentiment weighting

    Parameters
    ----------
            bow: list
                    a preprocessed bag of words to be checked against the topic model

            topicModel: gensim.models.ldamodel
                    the topic model being used to check the topics of a bag of words

    Output
    ------
            list
                    the topic distribution as a list of (topic number, weight) pairs, where
                    the topic number is an integer, and the weight is a float
    '''
    return topicModel.get_document_topics(bow, minimum_probability=cfg['malletsettings']['minimum_probability'])

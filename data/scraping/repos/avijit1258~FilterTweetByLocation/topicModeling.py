from gensim import corpora
import gensim
import tweetPreprocessing
from gensim.models.coherencemodel import CoherenceModel


tweets = ['Our proposal for a digital contact tracing system #PanCast  that goes beyond smartphone-based solutions. It''s a joint collaboration with Gilles Barthe, @rdeviti, Peter Druschel, Deepak Garg,  @autreche, Pierfrancesco Ingo, @mattlentz_ & @bschoelkopf. http://pancast.mpi-sws.org', 'Sending good vibes to everyone fighting to make it through BFCM','I really dropped all my hobbies and talents in order to focus on ‘productivity’ and chasing the bag in college damn']

def topic_model_lda(texts):
    ''' mine topics using LDA
        returns list(topic) of list (terms)
    '''
    NUM_TOPICS = 5
    NUM_WORDS = 7
    all_tokens = []

    for text in texts:
        tokens = tweetPreprocessing.preprocess_text(text)
        
        all_tokens.append(tokens)

    dictionary = corpora.Dictionary(all_tokens)
    corpus = [dictionary.doc2bow(token) for token in all_tokens]

    # ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=12)
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, alpha = 'auto', eval_every = 5)

    topics = []
    
    for topicId in range(NUM_TOPICS):
        topic = ldamodel.show_topic(topicId, topn = NUM_WORDS)
        topic = [t[0] for t in topic]
        topics.append(topic)

    # print(ldamodel.top_topics(corpus, dictionary, topn = NUM_WORDS))

    # topic coherence
    cm = CoherenceModel(model=ldamodel, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()  # get coherence value
    print('Coherence ', coherence)

    return topics


def test_topic_model_lda():
    print('...Testing topic_model_lda....')
    
    print(topic_model_lda(tweets))

    return 

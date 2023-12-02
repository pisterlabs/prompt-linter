import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

def build_model(data, num_topics, include_vis = True):
    '''
    Trains LDA model using Gensim based on parameters
    and saves a pyLDAvis visualization of the topics.

    :param data: list of tweets to be processed
    :param num_topics: number of topics for the model
    :param include_vis: flag to include pyLDAvis
    :return: LDA model as given by Gensim
    '''
    # Create Dictionary
    id2word = corpora.Dictionary(data)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

    #pyLDAvis
    if include_vis:
        p = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(p, 'lda_{}.html'.format(num_topics))

    return lda_model

def get_best(data, lo, hi, step):
    '''
    Trains LDA for varied number of topics and provides coherence information for each to optimize
    for number of topics.

    :param data: list of tweets to be processed
    :param lo: lower bound of topic number to consider
    :param hi: upper bound of topic number to consider
    :param step: step size for topic number iteration
    :return: saves plot of coherence scores for each topic model and csv of coherence scores
    '''

    tweets_coherence = []
    # Create Dictionary
    id2word = corpora.Dictionary(data)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data]
    for nb_topics in range(lo, hi, step):
        lda_model = build_model(data, num_topics = nb_topics, include_vis = False)
        cohm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence='u_mass')
        coh = cohm.get_coherence()
        tweets_coherence.append(coh)

    # visualize coherence
    plt.figure(figsize=(10, 5))
    plt.plot(range(lo, hi, step), tweets_coherence)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score");
    plt.savefig('lda_coherence.png', bbox_inches='tight')

    df = pd.DataFrame(data={"num_topics": range(lo, hi, step), "coherence": tweets_coherence})
    df.to_csv("coherence.csv", sep=',',index=False)

def top_vocab (lda_model, num):
    '''
    Gets top vocabulary words for a given LDA model and saves to csv.

    :param lda_model: LDA model to be evaluated
    :param num: number of vocabulary words assigned to each topic
    :return: csv of top words per topic
    '''
    top_words_per_topic = []
    for t in range(lda_model.num_topics):
        top_words_per_topic.extend([(t,) + x for x in lda_model.show_topic(t, topn=num)])
    pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words_{}.csv".format(lda_model.num_topics))


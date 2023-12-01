import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gensim
from gensim import corpora, models, similarities
from gensim.models.hdpmodel import HdpModel

url_names = ['cnn', 'abc', 'fox', 'nyt', 'reuters', 'wapo', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'washtimes']


def gensim_hdp_lda(df, n_topics):
    df = df[pd.notnull(df['processed_text'])]
    documents = df['processed_text'].values.tolist()

    texts = [[word for word in document.split()] for document in documents]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1]
    for text in texts]

    from pprint import pprint  # pretty-printer
    # pprint(texts)

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    print('Working on HDP...')
    hdp = HdpModel(corpus, dictionary, K=15, T=150)

    print(len(hdp.print_topics(-1)))


    def topic_prob_extractor(hdp=None, topn=None):
        topic_list = hdp.show_topics(-1, topn)
        topics = [x[1] for x in topic_list]
        split_list = [x[1] for x in topic_list]
        weights = []
        words = []
        for lst in split_list:
            temp = [x.split('*') for x in lst.split(' + ')]
            weights.append([float(w[0]) for w in temp])
            words.append([w[0] for w in temp])
        sums = [np.sum(x) for x in weights]
        return pd.DataFrame({'topic_id' : topics, 'weight' : sums})

    topic_weights = topic_prob_extractor(hdp, 10000)

    plt.plot(topic_weights['weight'])
    plt.xlabel('Topic')
    plt.ylabel('Probability of Topic')
    plt.title('Probability of Topic using HDP-LDA')

    num_topics=40

    from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
    ldamodel = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)

    import pyLDAvis.gensim
    vis_data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.save_html(vis_data, '../plots/pyLDAvis_'+str(num_topics)+'topics_using_gensim.html')

    def evaluate_graph(dictionary, corpus, texts, limit):
        """
        Function to display num_topics - LDA graph using c_v coherence

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        limit : topic limit

        Returns:
        -------
        lm_list : List of LDA topic models
        c_v : Coherence values corresponding to the LDA model with respective number of topics
        """
        c_v = []
        lm_list = []
        for num_topics in range(1, limit):
            lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
            lm_list.append(lm)
            cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
            c_v.append(cm.get_coherence())

        # Show graph
        x = range(1, limit)
        plt.plot(x, c_v)
        plt.xlabel("num_topics")
        plt.ylabel("Coherence score")
        plt.legend(("c_v"), loc='best')
        plt.show()

        return lm_list, c_v

        lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=train_texts, limit=10)


if __name__ == '__main__':
    df = pd.read_csv('../data/rss_feeds_new_data.csv')

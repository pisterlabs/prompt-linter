
from preprocessing import Pulizia
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import matplotlib.pyplot as plt
from preprocessing_es import *

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
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
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def graph_coherence(coherence_values, limit, start=2, step=1):
    # Show graph
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

if __name__ == '__main__':
    # lettura dataset
    df = pd.read_csv("dataset/Dataset.csv", error_bad_lines=False, sep=',')

    '''Coherence Dataset Totale '''
    df = df.loc[(df['Lingua'] == 'en')]
    df = df[0:100]

    '''Coherence Latin e lingua spagnola'''
    # df = df.loc[(df['Lingua'] == 'es') & (df['Genere'] == 'Latin')]
    # data_classes = ['Latin']

    testo = df['Testo']
    data_ready = Pulizia(testo)

    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_ready, start=2, limit=8, step=1)

    # Show graph
    limit = 8
    graph_coherence(coherence_values, limit, start=2, step=1)

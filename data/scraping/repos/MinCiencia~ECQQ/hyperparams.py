import gensim
from gensim.models import CoherenceModel, LdaMulticore



def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3, a=0.01, b=0.1):
    """
    Compute coherence values for a given number of topics between start and limit.

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of trained LDA models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, id2word=dictionary,
                             num_topics=num_topics, random_state=100,
                             chunksize=100, passes=10, alpha=a, eta=b, per_word_topics=True)

        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

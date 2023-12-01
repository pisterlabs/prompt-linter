from gensim.models import CoherenceModel


"""
Compute c_v coherence for various number of topics

Returns:
-------
model_list : List of LDA topic models
coherence_values : Coherence values corresponding to the LDA model with respective number of topics
"""
def compute_coherence_values(limit, start, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit+1, step):
        model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, random_state=101,passes=2, workers=4)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_values.append(coherence_score)
    return model_list, coherence_values
from gensim.models import CoherenceModel, LdaModel

def topic_num_search(tokens, corpus, id2word, min_topics, max_topics):
    """
    Calculates coherence for range of topics min to max
    """
    metrics_list = []
    
    for i in range(min_topics, max_topics + 1):
        
        lda = LdaModel(corpus=corpus, num_topics=i, id2word=id2word, chunksize=500, passes=40,
                   update_every=1, alpha="auto", eta="auto", random_state=137)

        x = lda.log_perplexity(corpus, total_docs=10000)
        
        lda_coherence = CoherenceModel(model=lda, texts=tokens, dictionary=id2word, coherence="c_v")
        y = lda_coherence_score = lda_coherence.get_coherence()

        metrics_list.append((i, x, y))
        
    return metrics_list
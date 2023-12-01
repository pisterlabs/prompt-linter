import numpy as np
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import ldamulticore
import multiprocessing as mp

DEFAULT_WORKERS = max(1, mp.cpu_count() - 1)

def create_id2word(tokenized_docs, filter_no_below=10, filter_no_above=0.5):
    id2word = corpora.Dictionary(tokenized_docs)
    id2word.filter_extremes(no_below=filter_no_below, no_above=filter_no_above)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in tokenized_docs]
    return id2word, corpus

def topic_model(tokenized_docs, num_topics=10, iterations=50, passes=10,
                chunksize=2000, workers=DEFAULT_WORKERS, **kwargs):
    id2word, corpus = create_id2word(tokenized_docs)

    model = ldamulticore.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        workers=workers,
        iterations=iterations,
        passes=passes,
        chunksize=chunksize,
        eval_every=10, # Setting this to one slows down training by ~2x
        per_word_topics=True)

    # computing perplexity and coherence
    perplexity = model.log_perplexity(corpus)
    coherence_model = CoherenceModel(model=model, texts=tokenized_docs, dictionary=id2word, coherence='c_v')
    coherence= coherence_model.get_coherence()
    return model, corpus, coherence, perplexity

def topic_vector(model, doc):
    num_topics = model.num_topics
    if not doc:
        return [0.] * num_topics
    corpus = model.id2word.doc2bow(doc.split())
    # https://radimrehurek.com/gensim/models/ldamulticore.html#gensim.models.ldamulticore.LdaMulticore.get_document_topics
    topics = model.get_document_topics(corpus, minimum_probability=0.0)
    return np.array([topics[i][1] for i in range(num_topics)])

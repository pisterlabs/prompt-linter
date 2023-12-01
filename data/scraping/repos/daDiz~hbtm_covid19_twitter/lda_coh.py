import gensim
import numpy
import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora

df = pd.read_csv('../data/tweets_cutoff0.01_sorted.csv',sep='\t')


texts = df['text'].values
texts = [x.split(' ') for x in texts]


id2word = corpora.Dictionary(texts)

corpus = [id2word.doc2bow(text) for text in texts]

num_topics=[2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
coh_scores = []
for ntop in num_topics:
    lda_model = LdaModel(corpus=corpus,
                           id2word=id2word,
                           num_topics=ntop,
                           random_state=42,
                           update_every=1,
                           chunksize=100,
                           passes=10,
                           alpha='auto',
                           per_word_topics=True)


    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_uci')

    coherence_lda = coherence_model_lda.get_coherence()

    coh_scores.append(coherence_lda)

df_out = pd.DataFrame.from_dict({'ntop':num_topics,'coh':coh_scores})

df_out.to_csv('lda_coh.csv',index=False)


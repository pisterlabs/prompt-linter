import mysql.connector
import pandas as pd
import settings
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from pprint import pprint
import numpy as np
import tqdm


db_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="YUrio_123",
        port="3306",
        database="twitterdb",
        charset = 'utf8'
    )

query = "SELECT * FROM {} where sentimen_tweet = 'positif'; ".format(settings.TABLE_NAME)
df_pos = pd.read_sql(query, con=db_connection)

query = "SELECT * FROM {} where sentimen_tweet = 'netral'; ".format(settings.TABLE_NAME)
df_net = pd.read_sql(query, con=db_connection)

query = "SELECT * FROM {} where sentimen_tweet = 'negatif'; ".format(settings.TABLE_NAME)
df_neg = pd.read_sql(query, con=db_connection)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_pos = df_pos.clean_tweet.values.tolist()
data_words_pos = list(sent_to_words(data_pos))

data_net = df_net.clean_tweet.values.tolist()
data_words_net = list(sent_to_words(data_net))

data_neg = df_neg.clean_tweet.values.tolist()
data_words_neg = list(sent_to_words(data_neg))

# Build the bigram and trigram models from positif sentimen
bigram_pos = gensim.models.Phrases(data_words_pos, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram_pos = gensim.models.Phrases(bigram_pos[data_words_pos], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram from positif sentimen
bigram_mod_pos = gensim.models.phrases.Phraser(bigram_pos)
trigram_mod_pos = gensim.models.phrases.Phraser(trigram_pos)

# Build the bigram and trigram models from netral sentimen
bigram_net = gensim.models.Phrases(data_words_net, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram_net = gensim.models.Phrases(bigram_net[data_words_net], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram from netral sentimen
bigram_mod_net = gensim.models.phrases.Phraser(bigram_net)
trigram_mod_net = gensim.models.phrases.Phraser(trigram_net)

# Build the bigram and trigram models from negatif sentimen
bigram_neg = gensim.models.Phrases(data_words_neg, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram_neg = gensim.models.Phrases(bigram_neg[data_words_neg], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram from negatif sentimen
bigram_mod_neg = gensim.models.phrases.Phraser(bigram_neg)
trigram_mod_neg = gensim.models.phrases.Phraser(trigram_neg)

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Form Bigrams from positif sentimen
data_words_bigrams_pos = make_bigrams(data_words_pos,bigram_mod_pos)
data_words_trigrams_pos = make_trigrams(data_words_pos,bigram_mod_pos,trigram_mod_pos)

# Form Bigrams from netral sentimen
data_words_bigrams_net = make_bigrams(data_words_net,bigram_mod_net)
data_words_trigrams_net = make_trigrams(data_words_net,bigram_mod_net,trigram_mod_net)

# Form Bigrams from negatif sentimen
data_words_bigrams_neg = make_bigrams(data_words_neg,bigram_mod_neg)
data_words_trigrams_neg = make_trigrams(data_words_neg,bigram_mod_neg,trigram_mod_neg)


# Create Dictionary from sentimen positif bigrams
id2word_bi_pos = corpora.Dictionary(data_words_bigrams_pos)
# Create Corpus from sentimen positif bigrams
texts_bi_pos = data_words_bigrams_pos
# Term Document Frequency from sentimen positif bigrams
corpus_bi_pos = [id2word_bi_pos.doc2bow(text) for text in texts_bi_pos]
# View from sentimen positif bigrams
print(corpus_bi_pos[:1][0][:30])


# Create Dictionary from sentimen positif trigrams
id2word_tri_pos = corpora.Dictionary(data_words_trigrams_pos)
# Create Corpus from sentimen positif trigrams
texts_tri_pos = data_words_trigrams_pos
# Term Document Frequency from sentimen positif trigrams
corpus_tri_pos = [id2word_tri_pos.doc2bow(text) for text in texts_tri_pos]
# View from sentimen positif trigrams
print(corpus_tri_pos[:1][0][:30])


# Create Dictionary from sentimen netral bigrams
id2word_bi_net = corpora.Dictionary(data_words_bigrams_net)
# Create Corpus from sentimen netral bigrams
texts_bi_net = data_words_bigrams_net
# Term Document Frequency from sentimen netral bigrams
corpus_bi_net = [id2word_bi_net.doc2bow(text) for text in texts_bi_net]
# View from sentimen netral bigrams
print(corpus_bi_net[:1][0][:30])

# Create Dictionary from sentimen netral trigrams
id2word_tri_net = corpora.Dictionary(data_words_trigrams_net)
# Create Corpus from sentimen netral trigrams
texts_tri_net = data_words_trigrams_net
# Term Document Frequency from sentimen netral trigrams
corpus_tri_net = [id2word_tri_net.doc2bow(text) for text in texts_tri_net]
# View from sentimen netral trigrams
print(corpus_tri_net[:1][0][:30])


# Create Dictionary from sentimen negatif bigrams
id2word_bi_neg = corpora.Dictionary(data_words_bigrams_neg)
# Create Corpus from sentimen negatif bigrams
texts_bi_neg = data_words_bigrams_neg
# Term Document Frequency from sentimen negatif bigrams
corpus_bi_neg = [id2word_bi_neg.doc2bow(text) for text in texts_bi_neg]
# View from sentimen negatif bigrams
print(corpus_bi_neg[:1][0][:30])

# Create Dictionary from sentimen negatif trigrams
id2word_tri_neg = corpora.Dictionary(data_words_trigrams_neg)
# Create Corpus from sentimen negatif trigrams
texts_tri_neg = data_words_trigrams_neg
# Term Document Frequency from sentimen negatif trigrams
corpus_tri_neg = [id2word_tri_neg.doc2bow(text) for text in texts_tri_neg]
# View from sentimen negatif trigrams
print(corpus_tri_neg[:1][0][:30])

#----------------------------------------------------------------------
# Build LDA model from sentimen positif bigrams
lda_model_bi_pos = gensim.models.LdaMulticore(corpus=corpus_bi_pos,
                                       id2word=id2word_bi_pos,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model_bi_pos.print_topics())
doc_lda_bi_pos = lda_model_bi_pos[corpus_bi_pos]

# Build LDA model from sentimen positif trigrams
lda_model_tri_pos = gensim.models.LdaMulticore(corpus=corpus_tri_pos,
                                       id2word=id2word_tri_pos,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model_tri_pos.print_topics())
doc_lda_tri_pos = lda_model_tri_pos[corpus_tri_pos]

#----------------------------------------------------------------------
# Build LDA model from sentimen netral bigrams
lda_model_bi_net = gensim.models.LdaMulticore(corpus=corpus_bi_net,
                                       id2word=id2word_bi_net,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model_bi_net.print_topics())
doc_lda_bi_net = lda_model_bi_net[corpus_bi_net]

# Build LDA model from sentimen netral trigrams
lda_model_tri_net = gensim.models.LdaMulticore(corpus=corpus_tri_net,
                                       id2word=id2word_tri_net,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model_tri_net.print_topics())
doc_lda_tri_net = lda_model_tri_net[corpus_tri_net]

#----------------------------------------------------------------------
# Build LDA model from sentimen negatif bigrams
lda_model_bi_neg = gensim.models.LdaMulticore(corpus=corpus_bi_neg,
                                       id2word=id2word_bi_neg,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model_bi_neg.print_topics())
doc_lda_bi_neg = lda_model_bi_neg[corpus_bi_neg]

# Build LDA model from sentimen negatif trigrams
lda_model_tri_neg = gensim.models.LdaMulticore(corpus=corpus_tri_neg,
                                       id2word=id2word_tri_neg,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model_tri_neg.print_topics())
doc_lda_tri_neg = lda_model_tri_neg[corpus_tri_neg]

#------------------------------------------------------------------
# Compute Coherence Score from bigram sentimen positif
coherence_model_lda_bi_pos = CoherenceModel(model=lda_model_bi_pos, texts=data_words_bigrams_pos, dictionary=id2word_bi_pos, coherence='c_v')
coherence_lda_bi_pos = coherence_model_lda_bi_pos.get_coherence()
print('Coherence Score: ', coherence_lda_bi_pos)

# Compute Coherence Score from trigram sentimen positif
coherence_model_lda_tri_pos = CoherenceModel(model=lda_model_tri_pos, texts=data_words_trigrams_pos, dictionary=id2word_tri_pos, coherence='c_v')
coherence_lda_tri_pos = coherence_model_lda_tri_pos.get_coherence()
print('Coherence Score: ', coherence_lda_tri_pos)

#------------------------------------------------------------------
# Compute Coherence Score from bigram sentimen netral
coherence_model_lda_bi_net = CoherenceModel(model=lda_model_bi_net, texts=data_words_bigrams_net, dictionary=id2word_bi_net, coherence='c_v')
coherence_lda_bi_net = coherence_model_lda_bi_net.get_coherence()
print('Coherence Score: ', coherence_lda_bi_net)

# Compute Coherence Score from trigram sentimen netral
coherence_model_lda_tri_net = CoherenceModel(model=lda_model_tri_net, texts=data_words_trigrams_net, dictionary=id2word_tri_net, coherence='c_v')
coherence_lda_tri_net = coherence_model_lda_tri_net.get_coherence()
print('Coherence Score: ', coherence_lda_tri_net)

#------------------------------------------------------------------
# Compute Coherence Score from bigram sentimen negatif 
coherence_model_lda_bi_neg = CoherenceModel(model=lda_model_bi_neg, texts=data_words_bigrams_neg, dictionary=id2word_bi_neg, coherence='c_v')
coherence_lda_bi_neg = coherence_model_lda_bi_neg.get_coherence()
print('Coherence Score: ', coherence_lda_bi_neg)

# Compute Coherence Score from trigram sentimen negatif
coherence_model_lda_tri_neg = CoherenceModel(model=lda_model_tri_neg, texts=data_words_trigrams_neg, dictionary=id2word_tri_neg, coherence='c_v')
coherence_lda_tri_neg = coherence_model_lda_tri_neg.get_coherence()
print('Coherence Score: ', coherence_lda_tri_neg)


def compute_coherence_values(corpus, dictionary, k, a, b,data_lemmatized,id2word):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()


def get_coherence_max(corpus_data,id2word,data_words,title_data):
  grid = {}
  grid['Validation_Set'] = {}

  # Topics range
  min_topics = 2
  max_topics = 11
  step_size = 1
  topics_range = range(min_topics, max_topics, step_size)

  # Alpha parameter
  alpha = list(np.arange(0.01, 1, 0.3))
  alpha.append('symmetric')
  alpha.append('asymmetric')

  # Beta parameter
  beta = list(np.arange(0.01, 1, 0.3))
  beta.append('symmetric')

  # Validation sets
  num_of_docs = len(corpus_data)
  corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
                 # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
                 # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75), 
                 corpus_data]

  corpus_title = [title_data]

  model_results = {'Validation_Set': [],
                   'Topics': [],
                   'Alpha': [],
                   'Beta': [],
                   'Coherence': []
                  }

  # Can take a long time to run
  if 1 == 1:
      pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))
      
      # iterate through validation corpuses
      for i in range(len(corpus_sets)):
          # iterate through number of topics
          for k in topics_range:
              # iterate through alpha values
              for a in alpha:
                  # iterare through beta values
                  for b in beta:
                      # get the coherence score for the given parameters
                      cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                    k=k, a=a, b=b,data_lemmatized=data_words,id2word=id2word)
                      # Save the model results
                      model_results['Validation_Set'].append(corpus_title[i])
                      model_results['Topics'].append(k)
                      model_results['Alpha'].append(a)
                      model_results['Beta'].append(b)
                      model_results['Coherence'].append(cv)
                      
                      pbar.update(1)
      pd.DataFrame(model_results).to_csv('lda_tuning_results_{}.csv'.format(title_data), index=False)
      pbar.close()

get_coherence_max(corpus_bi_pos,id2word_bi_pos,data_words_bigrams_pos,"sentimen_bi_pos")
get_coherence_max(corpus_tri_pos,id2word_tri_pos,data_words_trigrams_pos,"sentimen_tri_pos")

get_coherence_max(corpus_bi_net,id2word_bi_net,data_words_bigrams_net,"sentimen_bi_net")
get_coherence_max(corpus_tri_net,id2word_tri_net,data_words_trigrams_net,"sentimen_tri_net")

get_coherence_max(corpus_bi_neg,id2word_bi_neg,data_words_bigrams_neg,"sentimen_bi_neg")
get_coherence_max(corpus_tri_neg,id2word_tri_neg,data_words_trigrams_neg,"sentimen_tri_neg")

# ### Final Model Training
# 
# Based on external evaluation (Code to be added from Excel based analysis), train the final model

lda_model = gensim.models.LdaMulticore(corpus=corpus_bi_pos,
                                           id2word=id2word_bi_pos,
                                           num_topics=8, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha="asymmetric",
                                           eta=0.61)


# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus_bi_pos]


import pyLDAvis.gensim
import pickle 
import pyLDAvis

# Visualize the topics
pyLDAvis.enable_notebook()

LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus_bi_pos, id2word_bi_pos)

LDAvis_prepared






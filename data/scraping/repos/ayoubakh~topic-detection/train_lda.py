# gensim packages
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import numpy as np
import streamlit as st

def sent_to_words(sentences):
	for sentence in sentences:
		# deacc=True removes punctuations
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def prepare_training_data(tweets):
	data = tweets.clean_tweet.values.tolist()
	data_words = list(sent_to_words(data))
	# Create Dictionary
	id2word = corpora.Dictionary(data_words)
	# Create Corpus
	texts = data_words
	# Term Document Frequency
	corpus = [id2word.doc2bow(text) for text in texts]

	return id2word, corpus



def train_lda(tweets, parameters):
	id2word, corpus = prepare_training_data(tweets)

	num_topics, chunksize, alpha, beta, passes, iterations = parameters['num_topics'], parameters['chunksize'], parameters['alpha'], \
	                                                         parameters['beta'], parameters['passes'], parameters['iterations']

	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
	                                            id2word=id2word,
	                                            num_topics=num_topics,
	                                            chunksize=chunksize,
	                                            alpha=alpha,
	                                            eta=beta,
	                                            passes=passes,
	                                            iterations=iterations)

	return id2word, corpus, lda_model



def calculate_perplexity(model, corpus):
	return np.exp2(-model.log_perplexity(corpus))


def calculate_coherence(model, data_words, id2word, coherence):
	return CoherenceModel(model=model, texts=data_words, dictionary=id2word, coherence=coherence).get_coherence()

def train_lda_iterative (tweets, parameters):
	id2word, corpus = prepare_training_data(tweets)

	start, end, step, chunksize, alpha, beta, passes, iterations = parameters['min_num_topics'], parameters['max_num_topics'], \
		parameters['step'], parameters['chunksize'], parameters['alpha'], parameters['beta'], parameters['passes'], \
		parameters['iterations']

	models, coherence_s = [], []
	data = tweets.clean_tweet.values.tolist()
	data_words = list(sent_to_words(data))
	progress = st.markdown("")
	for num_topics in range(start, end + 1, step):
		progress.markdown(f"training on {num_topics} topics")
		lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
				id2word=id2word,
				num_topics=num_topics,
				chunksize=chunksize,
				alpha=alpha,
				eta=beta,
				passes=passes,
				iterations=iterations)
		models.append(lda_model)
		coherence_s.append(calculate_coherence(lda_model, data_words, id2word, "c_v"))
		pass
	progress.empty()
	return id2word, corpus, models, coherence_s
import datetime
import pandas as pd

from gensim import matutils
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldaseqmodel import *
from gensim import utils as gensim_utils
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

from sklearn.feature_extraction.text import CountVectorizer
from text_preprocess.utils_preprocess import *


def get_data(df, start_date, end_date, date_col='created_ts', rm_duplicate=True):
	df = df[df[date_col].between(start_date, end_date)]
	df['created_ts'] = pd.to_datetime(df['created_ts']).dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
	df['created_dt'] = df['created_ts'].dt.date
	df['all_text'] = (df['text'] + df['quoted_text'].fillna('')).str.lower()
	df.dropna(subset=['all_text'],inplace=True)
	# can't do drop_duplicates on a series of lists so use tuples
	if rm_duplicate:
		df.drop_duplicates(subset=['all_text'], inplace=True)
#	df['hashtags'] = df[text_col].apply(get_hashtags)
	df['hashtags'] = df.entities.apply(parse_hashtags)
	return df

def vectorize_text(df):
	cv = CountVectorizer(preprocessor=preprocessor,
						 stop_words=stop,
						 lowercase=True,
						 decode_error='replace',
						 ngram_range=(1, 2),
						 max_df=0.9,
						 min_df=10,
						 max_features=10000)
	sparse_df = cv.fit_transform(df['all_text'])
	gensim_df = matutils.Sparse2Corpus(sparse_df.T)

	vocab = cv.get_feature_names()
	id2word = {i: s for i, s in enumerate(vocab)}
	dictionary = Dictionary.from_corpus(gensim_df, id2word)  # for coherence model
	return gensim_df, id2word, dictionary


def vectorize_hashtags(df):
	cv = CountVectorizer(tokenizer=lambda x:x if x else '',
						 stop_words=[''],
						 max_df=1.0,min_df=5,
						 lowercase=False)
	hashtag_mat = cv.fit_transform(df['hashtags'])
	hashtag_voc = np.array(cv.get_feature_names())
	return hashtag_mat, hashtag_voc


def reparam_topics(old_model, ntopics, new_id2word, new_date_label):
	"""return the sstats from an old model shoehorned into the shape of the new model
	if ntopics is smaller than the old model, drop the least frequent topics
	if it is larger, add new topics and seed with some small value
	map the old words onto the new terms
	return the new parameters and the mapping between old and new topics"""
	# initialize sstats of shape[len(new_id2word), ntopics]
	# self.random_state.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
	random_state = gensim_utils.get_random_state(None)
	new_sstats = random_state.gamma(100., 1. / 100., (len(new_id2word), old_model.sstats.shape[1]))
	topics = old_model.topic_names

	# map common words between old and new vocabularies
	old_dict = {v: k for k, v in old_model.id2word.items()}
	new_dict = {v: k for k, v in new_id2word.items()}
	common_words = filter(old_dict.has_key, new_dict.keys())
	common_words.sort()
	common_old_keys = [old_dict.get(i) for i in common_words]
	common_new_keys = [new_dict.get(i) for i in common_words]
	new_sstats[common_new_keys, :] = old_model.sstats[common_old_keys, :]

	# remove or add topics (columns)
	to_add = ntopics - new_sstats.shape[1]
	if to_add > 0:
		new_sstats = np.hstack((new_sstats,
								random_state.gamma(100., 1. / 100., (len(new_id2word), to_add))))
		new_topics = [new_date_label + ":" + str(i) for i in xrange(to_add)]
		topics = old_model.topic_names + new_topics
	elif to_add < 0:
		worst = old_model.topic_freq.iloc[old_model.topic_freq.shape[0] - 1, :].argsort()[range(-to_add)].tolist()
		new_sstats = np.delete(new_sstats, worst, axis=1)
		topics = [i for j, i in enumerate(old_model.topic_names) if j not in worst]
	#		worst = old_model.topic_freq.argsort()[:to_add]
	#		mask = np.delete(np.arange(new_sstats.shape[1]), worst)
	#		new_sstats = new_sstats[:,mask]

	# check that sstats has shape[len(new_id2word), ntopics]
	assert new_sstats.shape[1] == ntopics
	assert new_sstats.shape[0] == len(new_id2word)
	return new_sstats, topics


# vanilla LDA  (with multicore support)
def fit_LdaMulticore(gensim_df, id2word,
				 num_topics, alpha, workers=None,
				 passes=1, iterations=1000,
				 update_every=1000, chunksize=1000,
				 minimum_topic_probability = 0.05, forget_weight=0.5,
				 random_state=0):
	model = LdaMulticore(corpus = gensim_df,
						 id2word = id2word,
						 num_topics = num_topics,
						 alpha = alpha,
						 workers = workers,
						 passes = passes,  # epochs
						 iterations = iterations,
						 chunksize = chunksize,  #batch size
						 minimum_probability = minimum_topic_probability,
						 decay = forget_weight,
						 per_word_topics = True,
						 random_state = random_state)
	return model


# vanilla LDA (not multicore)
def fit_LdaModel(gensim_df, id2word,
				 num_topics, alpha,
				 passes=15, iterations=10000,
				 update_every=1000, chunksize=1000,
				 minimum_topic_probability = 0.05, forget_weight=0.5,
				 distributed = True):
	model = LdaModel(corpus = gensim_df,
					 id2word = id2word,
					 num_topics = num_topics,
					 alpha = alpha,
					 passes = passes,  # epochs
					 iterations = iterations,
					 update_every = update_every,  #batch size
					 chunksize = chunksize,  #batch size
					 minimum_probability = minimum_topic_probability,
					 decay = forget_weight,
					 per_word_topics = True,
					 distributed = distributed)
	return model


# dynamic LDA (needs to be manually parallelized)
# fit dtm
def fit_LdaSeqModel(gensim_df, id2word, dictionary,
					ndocs_per_time,
					ntopics=3, alpha=0.01, chain_variance=0.05,
					passes=1):
	model = LdaSeqModel(corpus=gensim_df,
						id2word=id2word,
						num_topics=ntopics,
						alphas=alpha,
						chain_variance=chain_variance,
						time_slice=ndocs_per_time,
						passes=passes, chunksize=100,
						initialize='gensim')

	coherence = []
	for i in range(model.num_time_slices):
		topics_dtm = model.dtm_coherence(time=i)
		cm_DTM = CoherenceModel(topics=topics_dtm, corpus=gensim_df, dictionary=dictionary, coherence='u_mass')
		coherence += [cm_DTM.get_coherence()]

	metrics = {}
	metrics['coherence'] = sum(coherence) / len(coherence)

	#	print("Mean coherence: {}".format(coherence_mean))

	return model, metrics


def doc_topic_dist(model):
	doc_topic = model.gammas/model.gammas.sum(axis=1)[:, np.newaxis]
	doc_topic_by_time = np.vsplit(doc_topic, np.cumsum(model.time_slice)[:-1])  # split by time window (days)
	return(doc_topic_by_time)


def top_words(lda, topic_names=None, topn=20):
	if not topic_names:
		topic_names = [k for k in range(lda.num_topics)]

	top_words = {}
	for k in range(lda.num_topics):
		top_words[topic_names[k]] = [lda.id2word[id] for id, dist in lda.get_topic_terms(k, topn=topn)]
	top_words_df = pd.DataFrame(top_words, columns=topic_names)
	return top_words_df


def top_docs(doc_topic, df, topn=20):
  top_docs_id = (-doc_topic).argsort(axis=0)[0:topn].T
  top_docs_df = df.iloc[top_docs_id.flatten()][['created_dt','user.name','all_text']]
  top_docs_df['topic'] = np.repeat(range(doc_topic.shape[1]),topn)
  return top_docs_df
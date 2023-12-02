import logging
import os
import re
import threading

import gensim
import math

import numpy as np
import pyLDAvis.gensim
from gensim import matutils
from gensim.models import CoherenceModel, LdaModel

from TopicModeling import preprocessing
from TopicModeling.config import NUM_TOPICS
from common.database import DataBase

# Constansts
MIN_PAPER_TOPIC_PROB_THRESHOLD = (1 / NUM_TOPICS) * 5

# The files
base_dir = os.path.join(os.path.dirname(__file__), 'modelfiles')
LDA_MODEL_FILE = os.path.join(base_dir, 'lda.model')
DTM_MODEL_FILE = os.path.join(base_dir, 'dtm.model')
ATM_MODEL_FILE = os.path.join(base_dir, 'atm.model')
SERIALIZATION_FILE = os.path.join(base_dir, 'atm-ser.model')
SPARSE_SIMILARITY_FILE = os.path.join(base_dir, 'sparse_similarity.index')
PAPER_TOPIC_MATRIX_FILE = os.path.join(base_dir, 'paper_topic_matrix.npy')
AUTHOR_TOPIC_MATRIX_FILE = os.path.join(base_dir, 'author_topic_matrix.npy')
YEAR_TOPIC_MATRIX_FILE = os.path.join(base_dir, 'year_topic_matrix.npy')
YEAR_AUTHOR_TOPIC_MATRIX_FILE = os.path.join(base_dir, 'year_author_topic_matrix.npy')
AUTHOR_SIMILARITY_MATRIX_FILE = os.path.join(base_dir, 'author_similarity_matrix.npy')

# The parameters for the models
passes = 20
eval_every = 0
iterations = 200

# Database
db = DataBase('dataset/database.sqlite')


def get_lda_coherence_scores(corpus, dictionary, _range=range(5, 100, 5)):
	"""Returns a list of coherence scores for different number of topics."""
	logging.info('Getting coherence scores from LDA models.')

	outputs = []

	# Loop over num_topics
	for i in _range:
		logging.info('Creating LDA model for num_topics={}.'.format(i))

		# Create the model
		model = LdaModel(corpus=corpus, id2word=dictionary, alpha='auto',
						 eta='auto', num_topics=i, passes=10,
						 eval_every=eval_every)

		# Save the model to a file
		model.save('{}-{}-{}'.format(LDA_MODEL_FILE, 'test', i))

		# Create coherence model
		cm = CoherenceModel(model, corpus=corpus, dictionary=dictionary,
							coherence='u_mass')
		ch = cm.get_coherence()

		logging.info('Coherence for {} topics: {}'.format(i, ch))

		# Add to output
		outputs.append(
			(i, ch, model.show_topics())
		)

	return outputs


def visualize_model(model, corpus, dictionary, port=8000):
	vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
	pyLDAvis.show(vis, port=port)


def visualize_models(model_list, corpus, dictionary):
	i = 0
	for m in model_list:
		t = threading.Thread(target=visualize_model, args=(m, corpus, dictionary, 8000+i,))
		t.start()


def get_perplexity(model, chunk):
	log_perplexity = model.log_perplexity(chunk)
	perplexity = math.pow(2, -log_perplexity)
	return perplexity


def get_paper_topic_probabilities_matrix(model, corpus, dictionary, docno_to_index):
	"""Returns matrix of paper x topic where the value is the probability
	that paper belongs to the topic.

	:param model: LDA model
	:type model: gensim.models.LdaModel
	:param corpus: The corpus
	:type corpus: gensim.corpora.MmCorpus
	:param dictionary: The dictionary
	:type dictionary: gensim.corpora.dictionary.Dictionary
	:param docno_to_index: Dictionary from paper id to index in corpus
	:type docno_to_index: dict
	"""

	if os.path.exists(PAPER_TOPIC_MATRIX_FILE):
		logging.info('Using cached version of paper topic matrix. ({})'.format(PAPER_TOPIC_MATRIX_FILE))
		matrix = np.load(PAPER_TOPIC_MATRIX_FILE)
	else:
		logging.info('Creating paper topic matrix.')

		# Get papers from database
		papers = db.get_all_papers()

		matrix = np.zeros(shape=(len(papers), model.num_topics))

		for _id, paper in papers.items():
			probs = model[corpus[docno_to_index[_id]]]
			for topic, prob in probs:
				matrix[docno_to_index[_id], topic] = prob

		np.save(PAPER_TOPIC_MATRIX_FILE, matrix)

	return matrix


def get_year_topic_matrix(paper_topic_matrix, docno_to_index):
	"""
	Returns 2 dimensional array of year x topic where each value is a score
	"""

	if os.path.exists(YEAR_TOPIC_MATRIX_FILE):
		logging.info('Using cached version of year topic matrix. ({})'.format(YEAR_TOPIC_MATRIX_FILE))
		matrix = np.load(YEAR_TOPIC_MATRIX_FILE)
	else:
		logging.info('Creating year topic matrix.')
		papers = db.get_all_papers()

		years_to_docs = {}
		for _id, paper in papers.items():
			if paper.year not in years_to_docs:
				years_to_docs[paper.year] = []
			years_to_docs[paper.year].append(_id)

		matrix = np.zeros(shape=(len(list(years_to_docs.keys())), paper_topic_matrix.shape[1]))

		for year, docs in years_to_docs.items():
			for d in docs:
				for topic, prob in enumerate(paper_topic_matrix[docno_to_index[d]]):
					matrix[year - 1987, topic] += prob

		np.save(YEAR_TOPIC_MATRIX_FILE, matrix)

	return matrix


def get_year_author_topic_matrix(paper_topic_matrix, docno_to_index, author2doc):
	"""
	Returns 3 dimensional array of year x author x topic where each value is a score
	"""

	if os.path.exists(YEAR_AUTHOR_TOPIC_MATRIX_FILE):
		logging.info('Using cached version of year x author x topic matrix. ({})'.format(YEAR_AUTHOR_TOPIC_MATRIX_FILE))
		matrix = np.load(YEAR_AUTHOR_TOPIC_MATRIX_FILE)
	else:
		logging.info('Creating year x author x topic matrix.')
		papers = db.get_all_papers()

		years_to_docs = {}
		for _id, paper in papers.items():
			if paper.year not in years_to_docs:
				years_to_docs[paper.year] = []
			years_to_docs[paper.year].append(_id)

		matrix = np.zeros(shape=(len(list(years_to_docs.keys())), len(list(author2doc.keys())), paper_topic_matrix.shape[1]))

		for year, y_docs in years_to_docs.items():
			y_docs = [docno_to_index[d] for d in y_docs]
			for author, a_docs in [(a, [docno_to_index[i] for i in docs]) for a, docs in author2doc.items()]:
				author_index = list(author2doc.keys()).index(author)
				docs_this_year_this_author = list(set(y_docs) & set(a_docs))

				for d in docs_this_year_this_author:
					matrix[year - 1987, author_index] += paper_topic_matrix[d]

		np.save(YEAR_AUTHOR_TOPIC_MATRIX_FILE, matrix)

	return matrix


def get_author_topic_probabilities_matrix(paper_topic_probabilities_matrix, author2doc, docno_to_index):
	"""Returns matrix of authors x topic where the value is the probability
	that author belongs to the topic.

	:param model: LDA model
	:type model: gensim.models.LdaModel
	:param author2doc: Dict
	:type author2doc: dict
	"""

	if os.path.exists(AUTHOR_TOPIC_MATRIX_FILE):
		logging.info('Using cached version of author topic matrix. ({})'.format(AUTHOR_TOPIC_MATRIX_FILE))
		matrix = np.load(AUTHOR_TOPIC_MATRIX_FILE)
	else:
		logging.info('Creating author topic matrix.')

		matrix = np.zeros(shape=(len(list(author2doc.keys())), paper_topic_probabilities_matrix.shape[1]))

		for i, (author, docs) in enumerate(author2doc.items()):
			probs = np.zeros(shape=(paper_topic_probabilities_matrix.shape[1]))
			for doc in docs:
				probs += paper_topic_probabilities_matrix[docno_to_index[doc]]
			probs = probs / len(docs)
			matrix[i] = probs

		np.save(AUTHOR_TOPIC_MATRIX_FILE, matrix)

	return matrix


def get_author2doc():
	"""Return dict with short author names as key and a list of doc ids as values"""

	# Get all papers
	papers = db.get_all()

	# Create doc to author dictionary
	author2doc = {}
	for _id, paper in papers.items():
		for author in paper.authors:
			name = preprocessing.preproccess_author(author.name)
			if name not in author2doc:
				author2doc[name] = []
			author2doc[name].append(_id)
	logging.info('Number of different authors: {}'.format(len(author2doc)))

	return author2doc


def get_lda_model(corpus, dictionary, num_topics):
	"""Create new model or use a cached one.

	:param corpus: The corpus
	:type corpus: gensim.corpora.MmCorpus
	:param dictionary: The dictionary
	:type dictionary: gensim.corpora.dictionary.Dictionary
	:param num_topics: When building the model, how many topics to use.
	:type num_topics: int
	:returns: The LDA model
	:rtype: gensim.models.LdaModel
	"""
	if os.path.exists(LDA_MODEL_FILE):
		logging.info(
			'Using cached version of LDA model. ({})'.format(LDA_MODEL_FILE))
		model = LdaModel.load(LDA_MODEL_FILE)
	else:
		logging.info('Building LDA model.')

		# Create the model
		model = LdaModel(corpus=corpus, id2word=dictionary, alpha='auto',
						 eta='auto', num_topics=num_topics, passes=passes,
						 eval_every=eval_every, iterations=iterations)

		# Save the model to a file
		model.save(LDA_MODEL_FILE)

	return model


def get_atm_model(corpus, dictionary, author2doc, docno_to_index, num_topics):
	"""Return atm model

	:param corpus: The corpus
	:type corpus: gensim.corpora.MmCorpus
	:param dictionary: The dictionary
	:type dictionary: gensim.corpora.dictionary.Dictionary
	:param author2doc: author to doc ids
	:type author2doc: dict
	:param docno_to_index: Dictionary from paper id to index in corpus
	:type docno_to_index: dict
	:param num_topics: When building the model, how many topics to use.
	:type num_topics: int
	"""

	if os.path.exists(ATM_MODEL_FILE):
		logging.info('Using cached version of ATM model. ({})'.format(ATM_MODEL_FILE))
		model = gensim.models.AuthorTopicModel.load(ATM_MODEL_FILE)
	else:
		logging.info('Building ATM model.')

		for a, docs in author2doc.items():
			for i, doc_id in enumerate(docs):
				author2doc[a][i] = docno_to_index[doc_id]
		logging.info('Number of different authors: {}'.format(len(author2doc)))

		# Create the model
		model = gensim.models.AuthorTopicModel(
			corpus,
			id2word=dictionary,
			num_topics=num_topics,
			author2doc=author2doc,
			alpha='auto',
			eta='auto',
			passes=passes,
			eval_every=eval_every,
			serialized=True,
			serialization_path=SERIALIZATION_FILE
		)

		# Save the model to a file
		model.save(ATM_MODEL_FILE)
	return model





#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Used to train LDA models and LDAMallet
"""

#######################################

import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim

# Enable logging for gensim - optional
import logging

from argparse import ArgumentParser
import os, codecs

__author__ = "Rochan Avlur Venkat"
#__copyright__ = ""
#__credits__ = [""]
#__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Rochan Avlur Venkat"
__email__ = "rochan170543@mechyd.ac.in"

#######################################

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def force_unicode(s, encoding='utf-8', errors='ignore'):
	"""
	Returns a unicode object representing 's'. Treats bytestrings using the
	'encoding' codec.

	Arguments
	---------
	s: string to be encoded
	encoding: encoding type, defaults to `utf-8`
	errors: whether or not to ignore errors, defaults to `ignore`

	Returns
	---------
	unicode string
	"""

	if s is None:
		return ''

	try:
		if not isinstance(s, basestring,):
			if hasattr(s, '__unicode__'):
				s = unicode(s)
			else:
				try:
					s = unicode(str(s), encoding, errors)
				except UnicodeEncodeError:
					if not isinstance(s, Exception):
						raise
					# If we get to here, the caller has passed in an Exception
					# subclass populated with non-ASCII data without special
					# handling to display as a string. We need to handle this
					# without raising a further exception. We do an
					# approximation to what the Exception's standard str()
					# output should be.
					s = ' '.join([force_unicode(arg, encoding, errors) for arg in s])
		elif not isinstance(s, unicode):
			# Note: We use .decode() here, instead of unicode(s, encoding,
			# errors), so that if s is a SafeString, it ends up being a
			# SafeUnicode at the end.
			s = s.decode(encoding, errors)
	except UnicodeDecodeError, e:
		if not isinstance(s, Exception):
			raise UnicodeDecodeError (s, *e.args)
		else:
			# If we get to here, the caller has passed in an Exception
			# subclass populated with non-ASCII bytestring data without a
			# working unicode method. Try to handle this without raising a
			# further exception by individually forcing the exception args
			# to unicode.
			s = ' '.join([force_unicode(arg, encoding, errors) for arg in s])
	return s

def loadReviews(inp_path):

	readToken= []
	with codecs.open(inp_path, 'r', encoding='utf8') as File:
		for row in File:
			token_in_row = row.split(",")
			for i in range(len(token_in_row)):
				token_in_row[i] = force_unicode(token_in_row[i])
			readToken.append(token_in_row)
	return readToken

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=6):
	"""
	Compute c_v coherence for various number of topics

	Arguments
	---------
	dictionary: Gensim dictionary
	corpus: Gensim corpus
	texts: List of input texts
	limit: Max num of topics

	Return
	---------
	model_list: List of LDA topic models
	coherence_values: Coherence values corresponding to the LDA model with respective number of topics
	"""
	coherence_values = []
	model_list = []
	for num_topics in range(start, limit, step):
		model = gensim.models.ldamodel.LdaModel(corpus=corpus,
						id2word=id2word,
						num_topics=num_topics,
						random_state=100,
						update_every=1,
						chunksize=100,
						passes=10,
						alpha='auto',
						per_word_topics=True)
		model_list.append(model)
		coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
		coherence_values.append(coherencemodel.get_coherence())

	return model_list, coherence_values

if __name__ == '__main__':

	# Parse command-line arguments
	parser = ArgumentParser()
	parser.add_argument("-i", "--input-path",
			help="Path to parsed reviews", type=str)
	parser.add_argument("-o", "--output-path",
			help="Destination to save model", type=str)
	parser.add_argument("-s", "--score",
			help="Bool for calculating coherence score", type=bool, default=False)
	parser.add_argument("-t", "--topic-number",
			help="Number of topics to train the model for", type=int, default=100)
	parser.add_argument("-o", "--mallet-path",
			help="Path to mallet's bin file (ie. Mallet/bin/mallet)", type=str)
	args = parser.parse_args()

	# Define number of topics
	numTopics=args.topic_number

	# Load reviews into a list
	data_lemmatized = loadReviews(args.input_path)

	# Create Dictionary
	id2word = corpora.Dictionary(data_lemmatized)

	# Term Document Frequency
	corpus = [id2word.doc2bow(text) for text in data_lemmatized]

	# Build LDA model
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
						id2word=id2word,
						num_topics=numTopics,
						random_state=100,
						update_every=1,
						chunksize=100,
						passes=10,
						alpha='auto',
						per_word_topics=True)

	# LDAMallet
	mallet_path = args.mallet_path
	lda_model_mallet = gensim.models.wrappers.LdaMallet(mallet_path,
						corpus=corpus,
						num_topics=numTopics,
						alpha=65,
						id2word=id2word,
						workers=6,
						prefix=None,
						optimize_interval=10,
						iterations=5000,
						topic_threshold=0.0)

	lda_model.save(args.output_path + "LDA")
	lda_model_mallet.save(args.output_path + "LDA-Mallet")
	lda_model_mallet.print_topics(num_topics=-1, num_words=10)

	if args.score == True:
		# Compute Coherence Score
		coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
		coherence_lda = coherence_model_lda.get_coherence()
		print('\nCoherence Score: ', coherence_lda)

		start=10
		limit=100
		step=11

		model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start = 10, limit = 100, step = 11)
		x = range(start, limit, step)
		# Print the coherence scores
		for m, cv in zip(x, coherence_values):
			print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
import os
from searcher.es_search import SearchResults_ES
from searcher.corpus_manager import CorpusManager
#from searcher.models import QueryRequest, VisRequest
#from searcher.query_handler import QueryHandler
from searcher.corpus_manager import CorpusManager
from searcher.nlp_model_manager import NLPModelManager
from searcher.formatted_data_manager import FormattedDataManager
import pandas as pd
import numpy as np
import tqdm
import gensim
from gensim.models import CoherenceModel

STORAGE_PATH = "eval_data/lda_hyperparams/"

class ModelEvaluation:
	def __init__(self, query_obj):
		self.query_obj = query_obj
		self.dictionary = CorpusManager(self.query_obj)
		self.corpus = None
		self.raw_corpus = None
		self.df = None
		self.raw_doc_iter = SearchResults_ES(self.query_obj['database'], qry_obj=self.query_obj, rand=self.query_obj["rand"], cleaned=True)
		self.doc_iter = None

	def run_eval(self):
		if self.query_obj['method'] == "special":
			#unimpemented
			return
		else:
			self.run_lda_coherence()

	def single_lda_run(self, topics=None, a=0.91, b=.1, seed=100, savefile=None):
		ntopics = self.query_obj['num_topics'] if topics == None else topics
		if self.corpus == None:
			self.dictionary.create_ngrams()
			self.dictionary.create_dict()
			self.corpus = []
			self.raw_corpus = []

			self.doc_iter = SearchResults_ES(self.query_obj['database'], qry_obj=self.query_obj, rand=self.query_obj["rand"], tokenized=True, cm=self.dictionary)
			for x in self.raw_doc_iter:
				self.raw_corpus.append(x)
			for x in self.doc_iter:
				self.corpus.append(x)
		lda_model = gensim.models.LdaModel(
			corpus=self.corpus,
			id2word=self.dictionary.dct,
			num_topics=ntopics,
			random_state=seed,
			chunksize=100,
			passes=5,
			alpha=a,
			eta=b,
			eval_every=5,
			per_word_topics=True)

		topics = lda_model.show_topics(num_topics=ntopics, num_words=10, formatted=True)
		print("model alpha:{} beta:{} topics:{}".format(a, b, ntopics))
		print(topics)
		if savefile:
			lda_model.save(STORAGE_PATH + savefile)
		else:
			lda_model.save(STORAGE_PATH + 'test_model')


		perp = lda_model.log_perplexity(self.corpus)
		coherence_model_lda = CoherenceModel(model=lda_model, texts=self.raw_corpus, dictionary=self.dictionary.dct, coherence='c_v')
		with np.errstate(invalid='ignore'):
			lda_score = coherence_model_lda.get_coherence()
			print(lda_score)
			print(perp)
			return [lda_score, perp]

	def run_lda_coherence(self, min_topics=15, max_topics=40, step_size=5, test=False):
		if self.corpus == None:
			self.dictionary.create_ngrams()
			self.dictionary.create_dict()
			self.corpus = []
			self.raw_corpus = []

			self.doc_iter = SearchResults_ES(self.query_obj['database'], qry_obj=self.query_obj, rand=self.query_obj["rand"], tokenized=True, cm=self.dictionary)

			self.raw_doc_iter = SearchResults_ES(self.query_obj['database'], qry_obj=self.query_obj, rand=self.query_obj["rand"], cleaned=True, cm=self.dictionary)

			for x in self.raw_doc_iter:
				self.raw_corpus.append(x)
			for x in self.doc_iter:
				self.corpus.append(x)

		print("CORPUS")
		print(self.corpus)
		print("RAW_CORPUS")
		print(self.raw_corpus)


		step_size = step_size
		topics_range = range(min_topics, max_topics, step_size)
		# Alpha parameter
		alpha = list(np.arange(0.01, 1, 0.3))
		alpha.append('symmetric')
		alpha.append('asymmetric')
		# Beta parameter
		beta = list(np.arange(0.01, 1, 0.3))

		

		beta.append('symmetric')
		beta.append('auto')
		if test == True:
			alpha = [0.31]
			beta = [0.91]

		id2word = self.dictionary

		corpus_sets = [""]

		corpus_title = ['100% Corpus']

		model_results = {'Validation_Set': [],
					 'Topics': [],
					 'Alpha': [],
					 'Beta': [],
					 'Coherence': [],
					 'Perplexity' : []
					}
		total_count = len(alpha) * len(beta) * ((max_topics - min_topics) / step_size)
		print(total_count)
		if 1 == 1:
			pbar = tqdm.tqdm(total=total_count)
			# iterate through validation corpuses
			for i in range(len(corpus_sets)):
				# iterate through number of topics
				for k in topics_range:
					# iterate through alpha values
					for a in alpha:
						# iterare through beta values
						for b in beta:
							# get the coherence score for the given parameters
							count = 0
							cv = self.single_lda_run(topics=k, a=a, b=b)
							model_results['Validation_Set'].append(corpus_title[i])
							model_results['Topics'].append(k)
							model_results['Alpha'].append(a)
							model_results['Beta'].append(b)
							model_results['Coherence'].append(cv[0])
							model_results['Perplexity'].append(cv[1])
							pbar.update(1)
		self.df = pd.DataFrame(model_results)
		pbar.close()
		return 

	def write_eval(self, writefile=None):
		if writefile:
			self.df.to_csv(writefile, index=False)
		else:
			return self.df



if __name__ == '__main__':
	"""test_qry_obj = {'start': '1809', 'end': '2017', 'f_start': '-1', 'f_end': '-1', 'qry': '', 'maximum_hits': '10', 'method': 'multilevel_lda', 'stop_words': '', 'replacement': '', 'phrases': '', 'level_select': 'article', 'num_topics': 10, 'passes': '20', 'database': 'CaseLaw_v2', 'journal': 'all', 'jurisdiction_select': 'all', 'auth_s': '', 'family_select': 'both', 'min_occurrence': '-1', 'max_occurrence': '-1', 'doc_count': '500', 'ngrams' : False, 'model_name' : 'test', 'rand' : True}
	"""
	test_qry_obj = {'start': 'year', 'end': 'year', 'f_start': '-1', 'f_end': '-1', 'qry': 'apple OR banana', 'maximum_hits': '100', 'method': 'multilevel_lda', 'stop_words': '', 'replacement': '', 'phrases': '', 'level_select': 'article', 'num_topics': 10, 'passes': '20', 'database': 'Pubmed', 'journal': 'all', 'jurisdiction_select': 'all', 'auth_s': '', 'family_select': 'both', 'min_occurrence': '-1', 'max_occurrence': '-1', 'doc_count': '10', 'ngrams' : True, 'model_name' : 'test', 'rand' : False}

	me = ModelEvaluation(test_qry_obj)
	me.run_lda_coherence(test=True)
	me.write_eval(STORAGE_PATH + "test_pubmed_eval.csv")

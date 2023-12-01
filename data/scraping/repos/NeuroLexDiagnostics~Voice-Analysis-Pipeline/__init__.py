# -*- coding: utf-8 -*-
# from coherence import coherenceAnalisys
# -*- coding: utf-8 -*-
# This module implements the algorithm used in "Automated analysis of 
# free speech predicts psychosis onset in high-risk youths" 
# http://www.nature.com/articles/npjschz201530

import json
import sys
import numpy as np
import scipy
import os
import os.path

class lsaWrapperLocal():
	
	def __init__(self,corpus='en_tasa'):
		packipath = os.path.join('DigiPsych_API','lang_check','coherence_master')
		package_path = packipath
		path = {"en_tasa":"models/tasa_150"}
		path_models = path[corpus]
		dic_word2index = json.load( open( os.path.join( os.path.join(package_path,path_models), 'dictionary.json')) )
		self.dic_word2index= dict(zip(dic_word2index,range(len(dic_word2index))))
		self.dic_index2word= dict(zip(range(len(dic_word2index)),dic_word2index))
		self.u = np.load(os.path.join( os.path.join(package_path,path_models) , 'matrix.npy'))

	def get_vector(self,word, normalized=False,size=150):
		try: return self.u[self.dic_word2index[word],:][:int(size)]
		except: return np.zeros(size)[: int(size)]
	
	def index2word(self,i): 
		try: return self.dic_index2word[i]
		except: return None

	def word2index(self,w): 
		try: return self.dic_word2index[w]
		except: return None
		
	def _unitvec(self,v): return v/np.linalg.norm(v)
		
	def similarity(self,word1,word2,size=150): return np.dot( self._unitvec( self.get_vector(word1)) , self._unitvec( self.get_vector(word2))  ) 


class coherenceAnalisys():
	
	def __init__(self,corpus='en_tasa', dims=150 , word_tokenizer=lambda x: x.split(' ') , sentence_tokenizer=lambda txt: txt.split('.') ):
		self.corpus =  lsaWrapperLocal(corpus=corpus)
		self.word_tokenizer= word_tokenizer
		self.sentence_tokenizer= sentence_tokenizer
	def _unitvec(self,v): return v/np.linalg.norm(v)	
	
	def analysis_text(self,text, max_order=10):
		sentences = self.sentence_tokenizer(text.lower())
		vectorized_sentences = [[ self.corpus.get_vector(w) for w in self.word_tokenizer(s) if np.linalg.norm(self.corpus.get_vector(w))>0]  for s in sentences]
		mean_and_len = [ (np.mean(vec_sent,0), len(vec_sent)) for vec_sent in vectorized_sentences ]
		try: mean_vectors_series , len_words_per_vectors = zip(*[ t for t in mean_and_len if t[1]>0])
		except: return {}
		m = np.array( list(map(self._unitvec, mean_vectors_series)))
		max_order = min(m.shape[0],max_order)
		similarity_matrix = np.dot(m,m.T)
		similarity_orders = [ np.diag(similarity_matrix,i) for i in range(1,max_order)]
		similarity_metrics = { 'order_'+str(i):self._get_statistics(s) for i,s in enumerate(similarity_orders) }
		
		normalized_coeff=[ list(map(np.mean,zip(len_words_per_vectors[:-i],len_words_per_vectors[i:]))) for i in range(1,max_order)]
		similarity_orders_normalized = [ s/ np.array(coeff_list) for s, coeff_list in zip(similarity_orders,normalized_coeff)]
		similarity_metrics_normalized = { 'normalized_order_'+str(i):self._get_statistics(s) for i,s in enumerate(similarity_orders_normalized) }
		
		similarity_metrics.update(similarity_metrics_normalized)
		similarity_metrics.update({ 'vector_serie_'+str(i):s for i,s in enumerate(similarity_orders)} )		
		
		return similarity_metrics
		
	def _get_statistics(self,s):
		res={'mean':np.mean(s),'std':np.std(s),'min':np.min(s),'max':np.max(s)}
		for i in range(0,110,10): res['percentile_'+str(i)]=np.percentile(s,i)
		return res
			
		
		

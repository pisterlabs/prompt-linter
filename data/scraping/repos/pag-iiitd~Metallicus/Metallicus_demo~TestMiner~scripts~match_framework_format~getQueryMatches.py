import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import gensim
from gensim import corpora,models
import pickle
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import random
import re	
import sys
from gensim.models import CoherenceModel
from clus_simmat import *
from nltk.stem.wordnet import WordNetLemmatizer
from MatchScoreCalculate import *
import pandas as pd
import os
import sqlite3
 
from sqlite3 import Error
 
#returns top k matches for a query from a relevant cluster

#Input: query doc, function signature #Output: Top Function ids from database
"""Approach: For a doc (only function explanation) pick closest clusters with scores
 at a threshold from lda model- append theme code based on file name of lda model. 
 For each picked cluster load sim_mat file.
 Given the sim_mat- pick top 50 from it from the filtered clusters for the given query
 For those picked top 50s, compute 3 scores between all possible pairs with query and predict
 from classifier using: https://scikit-learn.org/stable/modules/tree.html
 Can use prediction probablity to shortlist final matches and return their FIDs- how map FID- can run
 sql query on function sig, repo_path and docstring https://likegeeks.com/python-sqlite3-tutorial/?
 """
path_sep=os.path.sep
themeslist={'V':'Validator','U':'utility', 'N':'nlp'}

#for a given query returns FIDS of top matches
def getmatchIDS(query_full,querysig,expsig=None):
	query= get_cleaned_doc(query_full).split()#as tokens
	queryfname=get_cleaned_doc(' '.join(camel_case_split2(querysig.split('(')[0]))).split()#as tokens
	cids=returnclusters(query+queryfname, themeslist)
	try:
		con=sql_connection()
		sim,bow,dict,doc,func=load_simmat_bow(cids)
		filteredfids=picktop(con,query,queryfname,sim,bow,dict,doc,func)
		#print(filteredfids)
		sim_mat_file='fulldoclist_similarity_matrix.pkl'
		#sim_mat_file='combined3doclist_similaritymatrix_skipgram.pkl'
		
		with open(sim_mat_file, "rb") as myFile:
			similarity_matrix = pickle.load(myFile)
		dictionary=corpora.Dictionary.load('fulldoclist_dict.gensim')
		classifier=pickle.load(open('crosslib_tree_classification_model_balanced.sav', 'rb')) #v2=[sim:0.81121909 psim:0.0283848  fsim:0.16039611]
		finalfids=[]
		print("Query signature: ",querysig)
		for f in filteredfids:
			(sim,psim,fsim),mdoc,msig=compute3scores(con,f,query_full,querysig,similarity_matrix,dictionary)
			matchstatus,wtscore=classifymatch(classifier,sim,psim,fsim)
			if msig=='url(value, public=False)':
				print(f,sim,psim,fsim, wtscore,matchstatus,querysig,msig)
			if matchstatus and max(sim,psim,fsim)>=0.55 and wtscore>=0.40: #wtscore was earlier 0.47
				finalfids.append((f,wtscore,msig))
				if(expsig and expsig==msig):
					print('FOUND!')
				#print(f,wtscore, msig, sim,psim,fsim)
			if not matchstatus and max(sim,psim)>0.6 and max(psim,fsim)>0.6 and max(sim,fsim)>0.6 and sim<0.95:#any two high sim but categorized false match, less than 0.95 condition to avoid duplicates
				finalfids.append((f,wtscore,msig))
		finalfids=sorted(finalfids, key=lambda x: x[1],reverse=True)
		print("Preliminary shortlist of match candidates (func_id, match_score, signature):\n",finalfids)
		#print(binning(finalfids,similarity_matrix,dictionary,classifier,con))
		fids_info=list()
		for fid in finalfids:
			fids_info.append(getinfoforFid(fid[0],con,fid[1]))
		#print(finalfids,fids_info)
		pickle.dump(fids_info,open('fids_info.pkl', 'wb'))
		return fids_info
	except Error:
		print(Error)
	finally:
		con.close()

#assign bins to matches to further refine matching. Inputs fids list of fn_signature, repo_path,doc, test_fp, library_name	
def binning(finalfids,similarity_matrix,dictionary,classifier,con):		
	"""Pick each doc and iterate to find compute3scores
	pass two scores to classifier and assign bin.
	To assign bin - check if bin exists where query_fid already exists- 
	add match as finalfids like tuple to those bins, else create new bin and add both
	Note: one function may exist in multiple bins"""
	bins=list()#list of lists of fids
	for q in finalfids:
		qfid=q[0]
		(queryfid,qtest,qfname,qsig,qdoc)=getinfoforFid(qfid,con)
		for m in finalfids:
			if q[0]!=m[0]:#not same fids
				#(mfid,mtest,mfname,msig,mdoc)=getinfoforFid(m[0],con)
				(sim,psim,fsim),mdoc,msig=compute3scores(con,m[0],qdoc,qsig,similarity_matrix,dictionary)
				matchstatus,wtscore=classifymatch(classifier,sim,psim,fsim)
				#print(q[0],m[0],matchstatus,wtscore)
				if matchstatus and wtscore>0.6:
					bins_ind=checkbin(bins,qfid)
					if(len(bins_ind)>0):
						for i in bins_ind:
							if(m[0] not in bins[i]):
								bins[i].append(m[0])
					else:
						bins.append([qfid,m[0]])#add new bin
						#also need to add qfid to bins than already contain m[0]
						mbins_ind=checkbin(bins,m[0])
						if(len(mbins_ind)>0):
							for i in mbins_ind:
								if(qfid not in bins[i]):
									bins[i].append(qfid)
				"""else:#not a match
					if(len(checkbin(bins,qfid))==0):
						bins.append([qfid])
					if(len(checkbin(bins,m[0]))==0):
						bins.append([m[0]])"""
	return bins
	
#return list of bin indices if fid in those bins
def checkbin(bins, fid):
	bin_indx=list()
	for ind in range(len(bins)):
		if fid in bins[ind]:
			bin_indx.append(ind)
	return bin_indx
				
def sql_connection():
	try: 
		con = sqlite3.connect('crosslib_final_full_v2.db')
		#print('Connected to db')
		return con
	except Error:
		print(Error)

def sql_fetch(con,selectquery,paramtoquery):
	cursorObj = con.cursor()
	cursorObj.execute(selectquery,paramtoquery)
	#print('query run...')
	rows = cursorObj.fetchall() #returned as list of tuple of attributes
	#for row in rows:
	#	print(row)
	return rows

	
#return list of CIDs 
def returnclusters(qtokens,themeslist):
	cids=[]
	for t in themeslist:
		topclusters=[]
		#themecode=t[0].upper()#to append for CID later
		#load lda models and corresponding dict
		ldamodel=models.LdaModel.load('dataset_'+themeslist[t]+'_lda.gensim')
		dictionary=corpora.Dictionary.load('dataset_'+themeslist[t]+'_dict.gensim')
		topclusters=gettopmatch_clusters(ldamodel,dictionary,qtokens)
		for i in topclusters:
			cids.append(t+str(i))
	#print(cids)
	return cids
	
#fetch path to sim_mat and bow per cluster for given CID and load simmat, bow to call top 10	
def load_simmat_bow(cluster_list):
	loadedsim_mat=[]
	loadedbow=[]
	loadeddict=[]
	loadeddoc=[]
	loadedfunc=[]
	
	for cid in cluster_list:
		#sql_query='select sm_id from cluster_functions where cluster_id='+cid+' limit 1'
		#loadedsim_mat.extend(sql_fetch(con,sql_query))
		#print('sim_matrices/'+str(cid[0])+'_'+cid[1:]+'_'+'matrix.pkl')
		with open('sim_matrices/'+str(cid[0])+'_'+cid[1:]+'_'+'matrix.pkl', "rb") as myFile: # save grouping in a file to load later
			loadedsim_mat.append(pickle.load(myFile))
		with open('sim_matrices/'+str(cid[0])+'_'+cid[1:]+'_'+'bow.pkl', "rb") as myFile: # save grouping in a file to load later
			loadedbow.append(pickle.load(myFile))
		with open('sim_matrices/'+str(cid[0])+'_'+cid[1:]+'_'+'dict.pkl', "rb") as myFile: # save grouping in a file to load later
			loadeddict.append(pickle.load(myFile))
		with open('sim_matrices/'+str(cid[0])+'_'+cid[1:]+'_'+'origdoc.pkl', "rb") as myFile: # save grouping in a file to load later
			loadeddoc.append(pickle.load(myFile))
		with open('sim_matrices/'+str(cid[0])+'_'+cid[1:]+'_'+'origfunc.pkl', "rb") as myFile: # save grouping in a file to load later
			loadedfunc.append(pickle.load(myFile))

	#print(len(loadedbow))
	return (loadedsim_mat,loadedbow,loadeddict, loadeddoc,loadedfunc)

#shortlist from clusters for query and list of simmat, bow and dict
def picktop(con,qtokens,queryfname,simmat,bow,dict,origdoc,origfunc,scorethreshold=0.55):# was earlier 0.6
	"""termsim_index = WordEmbeddingSimilarityIndex(model.wv)
	>>> bow_corpus = [dictionary.doc2bow(document) for document in common_texts]--given
	>>> similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix
	>>> docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
	>>> query = 'graph trees computer'.split()  # make a query
	>>> sims = docsim_index[dictionary.doc2bow(query)]"""
	#dict=corpora.Dictionary()
	#loaded docs of combinedsheet sheet 1 content only- bow captures the description part only
	
	#enable this and disable below if want to selct from entire corpus instead of from clusters
	"""with open('combinedoclist.txt','rb') as myFile:
		orig_doclist=pickle.load(myFile)
	with open('combinedfnamelist.txt','rb') as myFile:
		orig_fname=pickle.load(myFile) #loads only fname not signature
	dict=corpora.Dictionary.load('fulldoclist_dict.gensim')
	with open('fulldoclist_bow.pkl','rb') as myFile:
		b=pickle.load(myFile)
	with open('fulldoclist_similarity_matrix.pkl','rb') as myFile:
		s=pickle.load(myFile)
	docsim_index = SoftCosineSimilarity(b, s, num_best=50)
	sims=docsim_index[dict.doc2bow(qtokens)]
	sim_clone=list(sims)
	fids=[]
	for doc in sim_clone:
		try:
			fsimscore=s.inner_product(dict.doc2bow(queryfname), dict.doc2bow(get_cleaned_doc(' '.join(camel_case_split2(orig_fname[doc[0]]))).split()), normalized=True)
			if (doc[1]<scorethreshold and fsimscore<scorethreshold):
				#print("***")
				sims.remove(doc)
		except:
			sims.remove(doc)
	for doc in sims:
		fids.extend(getFIDs(con,orig_doclist[doc[0]],orig_fname[doc[0]]))"""
	#print(origfunc)
	fids=[]	
	for (s,b,d,od,of) in zip(simmat,bow,dict,origdoc,origfunc):
		#print(len(od),len(of))
		
		docsim_index = SoftCosineSimilarity(b, s, num_best=2000)
		sims=docsim_index[d.doc2bow(qtokens)]#top 50 sim tuples- of doc index and score, per cluster
		sim_clone=list(sims)
		#print(sims)
		for doc in sim_clone:
			#print(of[doc[0]],doc[1])
			try:
				"""if('email(value' in of[doc[0]] or 'isValid(String email)' in of[doc[0]]):
					print(of[doc[0]],doc[0],doc[1],s)"""

				fsimscore=s.inner_product(d.doc2bow(queryfname), d.doc2bow(get_cleaned_doc(' '.join(camel_case_split2(of[doc[0]].split('(')[0]))).split()), normalized=True)
					
				if (doc[1]<scorethreshold and fsimscore<scorethreshold):
					sims.remove(doc)
					
			except:
				#print(s)
				sims.remove(doc)
				#print("---",of[doc[0]])
			#print(od[doc[0]],of[doc[0]])#,get_cleaned_doc(' '.join(camel_case_split2(of[doc[0]]))).split())
		for doc in sims:
			#print(of[doc[0]].split('(')[0],getFIDs(con,od[doc[0]],of[doc[0]]),od[doc[0]])
			fid=getFIDs(con,od[doc[0]],of[doc[0]])
			fids.extend(fid)
			#print('\n****')
	return list(set(fids))#returns FIDs of shortlisted docs

#returns list of fids having docstring and fsig as passed in parameter
def getFIDs(con, docstring, fsig):
	#print("select fn_id from functions where fn_signature=? and documentation=?",(fsig,docstring))
	return sql_fetch(con,"select fn_id from functions where fn_signature=? and documentation like ?",(fsig,'%'+docstring[:15]+'%'))
	#return sql_fetch(con,"select fn_id from functions where fn_signature=?",(fsig))
	
#takes fid of potential match and query,sig pair to compute 3 scores and passes to classifier
def compute3scores(con,fid,query,querysig,similarity_matrix,dictionary):
	mdoc,msig=sql_fetch(con,"select documentation,fn_signature from functions where fn_id=?",(fid))[0]
	#print(msig,end=' ')
	return calculate_doc_distance(query, querysig, mdoc, msig, similarity_matrix, dictionary),mdoc,msig

#return True if score classified as match with high probablity, false otherwise	
def classifymatch(classifier,sim,psim,fsim, probthreshold=0.61):#was 0.6 before
	#use wt score only for ranking final functions
	#print(classifier.feature_importances_)
	wt_score=classifier.feature_importances_[0]*sim+classifier.feature_importances_[1]*psim+classifier.feature_importances_[2]*fsim
	if sim>0.95:# ignore processing duplicates
		return False,wt_score
	x_test = pd.DataFrame([[sim,psim,fsim]], columns =['doc_sim','par_sim', 'func_sim'])
	match=classifier.predict(x_test)[0]
	match_prob=classifier.predict_proba(x_test)[0,0]#prob of classified as 0= match
	#print(x_test,match,match_prob)
	if match==0 and match_prob>=probthreshold:
		return True,wt_score
	elif match==1 and match_prob>=0.4:
		return True,wt_score
	else:
		return False,wt_score
	
def get_cleaned_doc(doc, lemmatize=True):
	doc=doc.replace('\n',' ')
	doc=treat_hyphen(doc)
	doc = drop_special_chars(drop_urls_filepath(drop_example(drop_tags(get_function_explanation(doc)))))
	cleaned_doc = ' '.join([' '.join(camel_case_split2(word)).lower() for word in doc.split() if len(word)>1 and word not in STOP_WORDS])
	return lemmatize_doc(cleaned_doc) if lemmatize else cleaned_doc	
	
#nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))	
"""Perform preprocessing on the documentation text to tokenize and ready to feed to ldamodel
Preprocessing includes: stopwords elimination, splitting(on _ - camelcase etc.), lemmatization"""
def prepare_text_for_lda(text):
	#print(text)
	text=text.replace('\n',' ')#extract only function description for topic modeling
	text=get_function_explanation(text)#extract only function description for topic modeling
	tokens = tokenize(text)
	tokens = [token for token in tokens if len(token) > 1]
	tokens = [token for token in tokens if token not in en_stop]
	tokens = [get_lemma2(token) for token in tokens]
	return tokens
	
"""Given a tokenized query,
returns the index of the top matched topics from the corpus"""
def gettopmatch_clusters(ldamodel,dictionary,qtokens=[], threshold=0.05): # earlier threshold was 0.2, optional to send query as text or as tokens
	new_doc_bow = dictionary.doc2bow(qtokens)
	#print(ldamodel.get_document_topics(new_doc_bow))
	doc_lda = ldamodel[new_doc_bow]   
	flag=False
	topgrp=[]
	for index, score in sorted(doc_lda, key=lambda tup: -1*tup[1]):
		if score>=threshold:
			if(not flag):
				top=score
				flag=True
			#if(top-score<0.2):
			topgrp.append(index) # return only the top match
		else:
			return topgrp# would be empty if no relevant cluster

	return topgrp 

#takes doc as documentation,functionName
def tokenize(doc):
	tokens=nltk.word_tokenize(doc)
	tcopy=list(tokens)
	for t in tcopy:
		tokens.remove(t)
		tokens.extend(camel_case_split(t))
	tokens=[token.lower() for token in tokens]
	return list(set(tokens))

"""Function to perform tokenization of a token such as perform camel case tokenization, split on hyphens,
   dots, underscores, slashes etc. and then return a new list of tokens obtained"""

def camel_case_split(identifier):
	identifier=identifier.replace('{','')
	identifier=identifier.replace('}','')
	
	x=identifier.split("-")
	#print x
	y=[]
	for j in x:
	 z=j.split('_')
	 y+=z
	x=y[:]
	y=[]
	for j in x:
	 z=j.split("/")
	 y+=z
	#print y
	x=[]
	for j in y:
	 y1=j.split(".")
	 x+=y1
	#print x
	y=[]
	for j in x:
	 y1=j.split("=")
	 y+=y1
	x=[]
	for j in y:
	 y1=j.split("(")
	 x+=y1
	matches=[]
	for j in y:
	 y1=j.split("#")
	 x+=y1
	for j in y:
	 y1=j.split("<")
	 x+=y1
	for j in y:
	 y1=j.split(">")
	 x+=y1
	for i in x:
	 matches +=(re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', i))
	#print matches
	matches1=[]
	for i in matches:
	 #print i.group(0)
	 matches1.append(i.group(0).lower())
	#print matches1
	return matches1
	
def get_lemma2(word):
	return WordNetLemmatizer().lemmatize(word)

#takes signature as input and retains only datatype(datatype expansion not needed)- for only java functions	
def remove_var_fromsig(sig):	
	comma_ind=list()
	flag=False # to capture if inside <>
	ctr=0
	for c in sig:
		if c=='<':
			flag=True
		elif c =='>':
			flag=False
		elif c==',' and not flag:
			comma_ind.append(ctr)
		ctr=ctr+1
	if '()' not in sig:# to avoid case of function() or with single parameter
		comma_ind.append(sig.rfind(')'))
		sig_processed=''
	else:
		sig_processed=sig
	ind=0;
	if len(comma_ind)==1:#for some exceptional cases like sig as func(java.lang.String...) due to some issue
		sig_processed=' '.join(sig.split(' ')[:-1])+')'
	else:
		for i in comma_ind:#TODO: check if 'final' needs to be removed as per soot's need
			sig_processed=sig_processed+' '.join((sig[ind:i].strip().split(' '))[:-1])+','
			ind=i+1
		sig_processed=sig_processed[:-1]+')'
	#print(sig,sig_processed)
	return sig_processed

#extract fn_signature, repo_path,doc, test_fp, library_name and process test_fp path as maintained locally
def getinfoforFid(fid,con,wtscore=-1):
	#for java func call remove var as well
	(sig,repo,doc,lib,test)=sql_fetch(con,"select fn_signature,repo_path,documentation,library_name,test_fp from functions where fn_id=?",(fid))[0]
	fname=sig.split('(')[0]
	if test==None:
		return (fid, test,"",sig,doc)
	#print(test)
	
	#We found cases of repo having code inside src and main/java folders only- May have to be altered if more repos added in future
	if '.java' in repo:#NOTE: test files hae been merged with src paths for java repos. 
		sig=remove_var_fromsig(sig)
		i=test.find('java/')
		if i ==-1:
			i=test.find('src/')
			if i==-1:
				i=test.find('test/')
				if i!=-1:
					i=i+5
			else:
				i=i+4
		else:
			i=i+5
		test=test[i:].replace('/',path_sep)
		test_name=test[:test.find('.java')].replace(path_sep,'.')
		#print(test,test_name)
		if wtscore==-1:
			return (fid, test,test_name,sig,doc)
		else:
			return (fid, test,test_name,sig,doc,wtscore)
	else:#indicating python #test_filepath processing
		if '/' in lib:
			lib=lib.split('/')[-1]#extract only project name
		test=test.replace('/',path_sep)
		test='python_repo'+path_sep+lib+'_repo'+path_sep+test
		if wtscore==-1:
			return (fid,test,fname,sig,doc)
		else:
			return (fid,test,fname,sig,doc,wtscore)
		

"""if __name__ == "__main__":
	
	df=pd.read_excel('TestSet.xlsx',sheet_name='Sheet1',header=0)
	for querysig,query,expsig,label in zip(df['QFunction_signature'], df['QDocumentation'], df['MFunction_signature'], df['Label']):
		if label==0:
			print('\n************\nQuery: '+querysig+' '+query)
			print('\n++++++++++++\nExpected: '+expsig)
			print('\n++++++++++++\nMatches found:')
			fid_info=getmatchIDS(query,querysig,expsig) # list of tuples with fid and info. (FID,path_to_testfile,testname_fname,sig_processed_unprocessed)
	#query='Checks if a field has a valid url address. '
	#query= "Return whether or not given value is a valid URL.		If the value is valid URL this function returns ``True``, otherwise	:class:`~validators.utils.ValidationFailure`.		This validator is based on the wonderful `URL validator of dperini`_.		.. _URL validator of dperini:	https://gist.github.com/dperini/729294		Examples::		>>> url('http://foobar.dk')	True		>>> url('ftp://foobar.dk')	True		>>> url('http://10.0.0.1')	True		>>> url('http://foobar.d')	ValidationFailure(func=url, ...)		>>> url('http://10.0.0.1', public=True)	ValidationFailure(func=url, ...)		.. versionadded:: 0.2		.. versionchanged:: 0.10.2		Added support for various exotic URLs and fixed various false	positives.		.. versionchanged:: 0.10.3		Added ``public`` parameter.		.. versionchanged:: 0.11.0		Made the regular expression this function uses case insensitive.		.. versionchanged:: 0.11.3		Added support for URLs containing localhost		:param value: URL address string to validate	:param public: (default=False) Set True to only allow a public IP address"
	#querysig='url(value, public=False)'
	#getmatchIDS(query,querysig)	
else:"""
query=sys.argv[1]
print("Query documentation:\n",query)
querysig=sys.argv[2]
getmatchIDS(query,querysig)
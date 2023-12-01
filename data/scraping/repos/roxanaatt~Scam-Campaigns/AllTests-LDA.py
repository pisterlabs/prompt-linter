from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora, models, interfaces
import scampDB
import codecs
import sys 
from nltk.corpus import stopwords
import numpy as np
from sklearn.decomposition import NMF
import logging,bz2
import re
from gensim.models.coherencemodel import CoherenceModel
import os

reload(sys)  
sys.setdefaultencoding('utf8')

###############  Connect to DB and extract all the comments for each number  #######################
db = scampDB.DB()
cur=db.get_cursor("info_cur",True)
cur1=db.get_cursor("info_cur",True)
res = cur.execute('''SELECT distinct sc_tellnum FROM tbl_scamcall order by sc_tellnum''')
numbers = cur.fetchall()
doc_set = []
result = [""]
final_doc = ""
for number in numbers:
	doc_set = []
	final_doc = ""
	res1 = cur1.execute('''SELECT distinct on (sc_comment) sc_comment,sc_calltype FROM tbl_scamcall WHERE sc_tellnum=%s order by sc_comment,sc_calltype''',(number))
	#doc_set.append(cur1.fetchall())
	doc_set = cur1.fetchall()
	for item in doc_set:
		for d in item:
			if d =='None':
				d = d.replace('None','')
			if d == None:
				continue
			d= d.decode('ascii','ignore')
			
			final_doc += " ".join(d.split()) + " "
	result.append(final_doc)
	#for item in doc_set:
	# doc_set_final = (str(doc_set)).replace('[','').replace(']','').replace('[\ ','').replace("\"",'').replace('\\','')#.replace('xc2xa0','')

	# #doc_set_final = re.sub('$xc2xa0\'', '', doc_set_final)
	# result.append(str(doc_set_final))


#### stop words out and stemming ########
texts = []
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
#p_stemmer = PorterStemmer()
#en_stop = set(stopwords.words('english'))
for doc in result:
	raw = doc.lower()
	tokens = tokenizer.tokenize(raw)
	stopped_tokens = [i for i in tokens if not i in en_stop]
	#stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	#texts.append(stemmed_tokens)
	texts.append(stopped_tokens)

######## Constructing a document-term matrix---- turn our tokenized documents into a id <-> term dictionary ###########
dictionary = corpora.Dictionary(texts)

######################### Prepared ####################
#convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

#to have logs
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


################# LDA #####################
#ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20,id2word=dictionary, update_every=1, chunksize=100, passes=1)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2,id2word=dictionary, minimum_probability=0.01)
ldamodel.print_topics(num_topics = 2, num_words = 20)


cm = CoherenceModel(model=ldamodel, corpus=corpus, dictionary=dictionary, coherence='u_mass') # note that a dictionary has to be provided.
print cm.get_coherence()

#print "...........................get__document_topics....................."

#print ldamodel.get_document_topics(corpus[100], minimum_probability=None)

# ################  NMF #####################

# model = NMF(n_components=2, init='random', random_state=0)
# model.fit(dictionary)
# NMF(alpha=0.0, beta=1, eta=0.1, init='random', l1_ratio=0.0, max_iter=200,
#   n_components=2, nls_max_iter=2000, random_state=0, shuffle=False,
#   solver='cd', sparseness=None, tol=0.0001, verbose=0) 







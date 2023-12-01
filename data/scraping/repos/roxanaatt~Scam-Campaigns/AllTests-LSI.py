from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import corpora, models, similarities
import scampDB
import codecs
import sys 
from nltk.corpus import stopwords
import numpy as np
from sklearn.decomposition import NMF
import logging,bz2
import re
from gensim import corpora, models, interfaces
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary


reload(sys)  
sys.setdefaultencoding('utf8')

###############  Connect to DB and extract all the comments for each number  #######################
db = scampDB.DB()
cur=db.get_cursor("info_cur",True)
cur1=db.get_cursor("info_cur",True)
cur2=db.get_cursor("info_cur",True)
res = cur.execute('''SELECT distinct sc_tellnum FROM tbl_scamcall order by sc_tellnum''')
numbers = cur.fetchall()
doc_set = []
result = [""]
final_doc = ""

for number in numbers:
	final_doc = ""
	doc_set = []
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

	
	cur2.execute('''INSERT INTO doc_content(content) VALUES (%s)''',(final_doc))
	db.commit()
	result.append(final_doc)
	

#### stop words out and stemming ########
texts = []
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
#p_stemmer = PorterStemmer()
#en_stop = set(stopwords.words('english'))
for doc in result:
	#raw = str(doc).lower().replace('[','').replace(']','')
	raw = doc.lower()

	tokens = tokenizer.tokenize(raw)
	stopped_tokens = [i for i in tokens if not i in en_stop]
	#stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	#texts.append(stemmed_tokens)
	texts.append(stopped_tokens)


######## Constructing a document-term matrix---- turn our tokenized documents into a id <-> term dictionary ###########
# The mapping between the questions and ids. Results is like {'comission':6625, ...}
dictionary = corpora.Dictionary(texts)

######################### Prepared ####################
#for each document, convert tokenized documents into a document-term matrix
#it simply counts the number of occurrences of each distinct word,in each document, converts the word to its integer word id and returns the result as a sparse vector
corpus = [dictionary.doc2bow(text) for text in texts]

######### TFIDF Model ###############
tfidf = models.TfidfModel(corpus) 
corpus_tfidf = tfidf[corpus]


#to have logs
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#################  LSA  ##################
#mm = gensim.corpora.MmCorpus('corpus')
#print mm
lsimodel = gensim.models.lsimodel.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
lsimodel.print_topics(num_topics = 200, num_words = 20)
#topics =  lsimodel.show_topics(num_topics=200, num_words=20, log=False, formatted=False)


############ Get the topic words from the model and build topics ###################
topics = []
for topic_id, topic in lsimodel.show_topics(num_topics=200,num_words = 20, formatted=False):
	topic = [word for word, _ in topic]
	topics.append(topic)

############# topic coherence ##################

cm = CoherenceModel(topics = topics , corpus=corpus, dictionary=dictionary, coherence='u_mass') # note that a dictionary has to be provided.
coherenceResult = cm.get_coherence()
print coherenceResult

#print "...........................get__document_topics....................."

# for corp in corpus:
# 	print "*****corpus = " + corp
# 	print "related topic = "
# 	print lsimodel.get_document_topics(corp,minimum_probability=None)


################## Send the result to my  Email#####################
# import smtplib
# msg = str(coherenceResult) + ' For 200 topics'

# server = smtplib.SMTP('smtp.gmail.com',587) #port 465 or 587
# server.ehlo()
# server.starttls()
# server.ehlo()
# server.login('example@gmail.com','password')
# server.sendmail('example@gmail.com','example@gmail.com',msg)
# server.close()






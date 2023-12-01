import nltk
import pymongo
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint  

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["travelbig1"]
mycol = mydb["big_col1"]
mycol3=mydb["topicwords"]
mycol4=mydb["doc_words"]

#mycol3 used for storing topic numbers and the words in eac topic
#mycol2 in travelbig2 database contains numbered documents. this db is to be used for displaying the results
#mycol4 is used to store each word and the document number to which it belongs

def process_text(query):
  stop_words = set(stopwords.words('english')) 

  query=re.sub(r"[^A-Za-z]"," ",query)

  word_tokens = word_tokenize(query.lower()) 

  filtered_sentence = [w for w in word_tokens if not w in stop_words] 

  
  return filtered_sentence

train_text=[]

docmap={}

i=0
for x in mycol.find():
	document=(x['text']).lower()
	temp=process_text(document)
	
	for word in temp:
		docmap[word]=i

		
	i=i+1
	train_text.append(temp)

print("no of entries in train text is %d"%len(train_text))

dictionary = Dictionary(train_text)

corpus = [dictionary.doc2bow(text) for text in train_text]


hdpmodel1 = HdpModel(corpus=corpus, id2word=dictionary)


x=hdpmodel1.show_topics(num_topics=30,num_words=200)


twords={}
for topic,word in x:
	twords[str(topic)]=(re.sub('[^A-Za-z ]+', '', word)).split()


mycol3.insert_one(twords)
mycol4.insert_one(docmap)



import nltk
# Create a corpus from a list of lists of tokens
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords

#Given an object named 'texts' (a list of tokenized texts, ie a list of lists of tokens)
#from gensim.test.utils import common_texts as texts


nltk.download('stopwords')

texts=[]
file = open("TokenVieuxM.txt", "r")
lines = file.readlines()
file.close()

#print(stopwords.words())

for line in lines:
  line=line.strip()
  lt=line.split(",")
#Potential ill-character cleaning
  for i in range(len(lt)):
    lt[i]=lt[i].replace('[','')
    lt[i]=lt[i].replace(']','')
    lt[i]=lt[i].replace('"','')
    lt[i]=lt[i].replace('\n','')
    lt[i]=lt[i].replace(' ', '')
#End : Potential ill-characters cleaning
# print(lt)
#  ltc=[word for word in lt if not word in stopwords.words()]
#  print("C", ltc)
  texts.append(lt)

#for text in texts:
  #print(text)

#Here set the number of topics(to be changed if necessary)
nb=10
  
id2word = Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]
#print(corpus)

# Print dictionnary
for i in id2word:
  print(i, id2word[i])

# Train the lda model on the corpus.
from gensim.models import LdaModel

lda = LdaModel(corpus, num_topics=nb)

# Print topic descrition
for i in range(0, nb-1):
  value=lda.get_topic_terms(i)
#  print(value)
  print("Topic ", i)
  for j in value:
    print(id2word[j[0]], " - P=", j[1])
  print()

# Compute Perplexity
perplexity_lda=lda.log_perplexity(corpus)  # a measure of how good the model is (lower the better).
print('Perplexity= ', perplexity_lda)

# Compute Coherence Score
from gensim.models import CoherenceModel

coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence= ', coherence_lda)


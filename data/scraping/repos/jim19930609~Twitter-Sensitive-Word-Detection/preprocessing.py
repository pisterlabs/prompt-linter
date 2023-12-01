import numpy as np
import os
import re
import nltk
import pyLDAvis.gensim
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from dataload import generate_training

def Remove_Symbols(name):
  rule = re.compile(r"[^a-zA-Z0-9 ]")
  new_name = rule.sub('', name)
  new_name = new_name.lower()
  
  return new_name

def collect_topics(lda_model, num_topics):
  f = open("results/topics.txt", "a")
  for topics in lda_model.print_topics(num_topics):
    star_sp = topics[1].split("*")
    for i in range(1, len(star_sp)):
      tp_word = star_sp[i].split("\"")[1]
      f.write(tp_word + " ")
    f.write("\n")

  f.close()

if __name__ == "__main__":
  path = "database/lda_training.csv"
  num_train = 30000
  num_add = 0
  num_topics = 80
  text_raw = generate_training(num_train, num_add, path)
  tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
  model_path = "lda_model/model"
  
  print ("Total Number of ", len(text_raw), " Training Data Loaded")

  # Load stop words
  words_stop = stopwords.words('english')
  stop_set = set()
  for i in range(len(words_stop)):
    stop_set.add(Remove_Symbols(words_stop[i]))


  text_split = []
  for i in range(len(text_raw)):
    word_nonstop = []
    for word in tknzr.tokenize(text_raw[i]):
      if word not in stop_set:
        if len(word) <= 3:
          continue
        
        word_nonstop.append(word)
    
    if len(word_nonstop) > 0:
      text_split.append(word_nonstop)

  # Word Dictionary
  dic = corpora.Dictionary(text_split)

  # Generate Corpus
  corpus = [dic.doc2bow(text) for text in text_split]

  if not os.path.exists(model_path):
    print ("Training LDA Model From Original")
    lda_model = models.ldamodel.LdaModel(corpus, id2word=dic, num_topics=num_topics, iterations=2000)
    lda_model.save(model_path)
  else:
    print ("Load Existing LDA Model")
    lda_model = models.ldamodel.LdaModel.load(model_path)

  # Make Predictions: 
  newdoc_count = corpus[0]
  doc_lda = lda_model[newdoc_count]
  
  collect_topics(lda_model, num_topics)

  cm = CoherenceModel(model=lda_model, texts=text_split, dictionary=dic, coherence='c_v')
  print (cm.get_coherence())
  
  data = pyLDAvis.gensim.prepare(lda_model, corpus, dic)
  pyLDAvis.show(data, open_browser=False)
  

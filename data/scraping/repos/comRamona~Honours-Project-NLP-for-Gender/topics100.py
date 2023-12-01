from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import MmCorpus
from gensim.corpora import Dictionary
from collections import Counter, defaultdict
from numpy.random import seed
from _storage.storage import FileDir
from os.path import join
import re
from metadata.metadata import ACL_metadata
from _topic_modeling.lda_loader import Loader
from _storage.storage import FileDir
from os.path import join
import _pickle as pkl
from metadata import Gender
import numpy as np
import logging
import gensim 

seed(1)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

def my_dist(v1,v2):
  v1 = list(map(lambda x: x[0], v1))
  v2 = list(map(lambda x: x[0], v2))
  d = 0
  n = len(v2)
  for i in v1:
      if i in v2:
          w = n - v2.index(i)
          d += w
          #print(d)
  return d


fd = FileDir()
loader = Loader()
lda = gensim.models.ldamodel.LdaModel.load(join(fd.models, "ldaseed310lda"))
with open(join(fd.models,"topic-terms.txt"),"r") as f:
    topics = f.read().split("\n")

j_doc_topics = defaultdict(list)
j_dic = gensim.corpora.Dictionary()
vv = list()
for topic in topics:
    if len(topic) > 10:
        number, name, words = topic.split(":")
        t = words.strip()
        tt = list(map(lambda x: x.strip(), re.split("\\[0\\.\\d\\d\\]",t)))
        if len(tt) == 151:
            tt = tt
            j_doc_topics[number + name] = tt
            j_dic.add_documents([tt])
            vv.append(tt)
j_topics_bow = defaultdict(list)
a = []
for i, jbow in j_doc_topics.items():
    j_topics_bow[i] = j_dic.doc2bow(jbow)
    a.append(j_dic.doc2bow(jbow))

r_doc_topics = defaultdict(list)
for t_n in range(100):
    for i, p in lda.get_topic_terms(t_n,151):
        r_doc_topics[t_n].append(loader.id2word[i])

corresp = defaultdict()
for i,r in r_doc_topics.items():
    r_2bow = j_dic.doc2bow(r)
    score = 0
    #sims = index[r_2bow]
    best_name = ""
    for name, jt in j_topics_bow.items():
        #result = gensim.matutils.cossim(r_2bow, jt)
        result = my_dist(r_2bow, jt)
        if(result > score):
            best_name = name
            score = result
    corresp[i] = best_name
    x = lda.show_topic(i, 10)
    words = ""
    for w, sc in x:
        words += (w + " ")
    print(i, best_name, words)


fd.save_pickle(corresp, "topic_corresp10")
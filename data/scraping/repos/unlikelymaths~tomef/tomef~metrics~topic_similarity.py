#!/usr/bin/env python
# coding: utf-8

# # Topic Similarity
# <div style="position: absolute; right:0;top:0"><a href="./metrics_index.doc.ipynb" style="text-decoration: none"> <font size="5">←</font></a>
# <a href="../evaluation.ipynb" style="text-decoration: none"> <font size="5">↑</font></a></div>
# 
# `Description`
# 
# ---
# ## Setup and Settings
# ---

# In[20]:


from __init__ import init_vars
init_vars(vars(), ('info', {}))

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import CoherenceModel, KeyedVectors
from gensim.corpora import WikiCorpus, Dictionary

import data
import config
from base import nbprint
from util import ProgressIterator
from widgetbase import nbbox
from os.path import join, isfile
from tokenizer.main import get_tokenizer

from metrics.widgets import topiclist_picker

if RUN_SCRIPT: topiclist_picker(info)


# In[24]:


def mean_pairwise_jaccard(topiclist):
    topiclist = [[entry.token for entry in topic] for topic in topiclist]
    similarities = []
    for idx, topic1 in enumerate(topiclist):
        set1 = set(topic1)
        for topic2 in topiclist[idx+1:]:
            set2 = set(topic2)
            similarities.append(len(set1.intersection(set2)) / len(set1.union(set2)))
    return sum(similarities) / len(similarities)


# ---
# ## Show all
# ---

# In[25]:


if RUN_SCRIPT:
    nbbox(mini=True)
    topiclist = data.load_topiclist(info)
    topiclist = [topic[:10] for topic in topiclist]
    mean_similarity = mean_pairwise_jaccard(topiclist)
    nbprint('Mean Pairwise Jaccard similarity is {}'.format(mean_similarity))


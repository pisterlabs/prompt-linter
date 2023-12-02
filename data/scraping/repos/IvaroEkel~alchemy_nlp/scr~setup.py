# libraries

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import sklearn as sk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Imports classifier model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# Creates confusion matrix and classification report using predictions for test data
from sklearn.metrics import confusion_matrix, classification_report
# Cross validation score
from sklearn.model_selection import cross_val_score
from statistics import mean
# from pprint import pprint
# import pyLDAvis
# import pickle 
# import pyLDAvis.sklearn

#usr/bin/python

# Packages

## standard library
import sys,os
sys.path.append(os.path.join(".."))
from pprint import pprint
from tqdm import tqdm
import numpy as np


## data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

## stopwords
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

## visualisation
import pyLDAvis.gensim
import pyLDAvis

## LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

## warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# Functions for functions
  
def data_wrapping():
  """
  collapses sentences tro full texts by title
  """
  data = pd.read_csv(os.path.join("..", "data", "interim","small_dataset.csv"))
    
  # group by title (author and school just to keep in, don't have any effect)
  df = data.groupby(["author","school","title"])

  # join all texts from one title into a single row
  df = df["sentence_str"].agg(lambda x : ' '.join(x)).to_frame()
    
  df = df.reset_index().rename(columns = {"sentence_str":"text"})
  print("Data has been loaded and collapsed by a book title.")
  return(df)


def divide_chunks(tokens, n):
  """
  divides a list of tokens into multiple lists of size n
  """
  # looping till length l
  for index in range(0,len(tokens), n): 
     yield tokens[index:index + n]


def chunking():
  """
  transforms dataframe of full texts into a list of chunked texts of 2000 tokens each
  """
  word_list = []
  chunk_list = []
  text_chunks = []
    
  # comma separating every word in a book
  for entry in range(len(df)):
      word_list.append(df.text[entry].split())
    
  # create a chunk of 2000 words
  for entry in word_list:
      chunk_list.append(list(divide_chunks(entry, 2000)))
    
  # flatten chunk list from a nested list to a list
  text_chunks = [item for l in chunk_list for item in l]
  
  print("Texts have been divided into chunks of 2000 tokens each for easier preprocessing")
  return(text_chunks)

## Ross' function for stopwords, lemmatization
def process_word_no_grams():
  """Remove Stopwords, lemmatize, and add pos-tags to lemmas"""

  # use gensim simple preprocess
  allowed_postags=['NOUN', "ADJ", "VERB"]
  text_chunks = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in tqdm(texts)]
  print("the texts have been roughly preprocessed")
  texts_out = []
  sentence_list = []
  # lemmatize and POS tag using spaCy
  print("lemmatizing and pos-tagging docs...")
  for sent in texts:
      sentence_list.append(" ".join(sent))
  
  for doc in tqdm(nlp.pipe(sentence_list, n_process = -1), total = len(sentence_list)):
      texts_out.append([f"{token.lemma_}_{token.pos_}" for token in doc if token.pos_ in allowed_postags]) 
  return texts_out

def lda_modeling():
  """
  Make a dictionary, run a model, calculate coherence and perplexity, save a vizualization
  """
  # Create Dictionary
  id2word = corpora.Dictionary(data_processed)
  id2word.filter_extremes(no_below=25)
  corpus = [id2word.doc2bow(text) for text in data_processed]
  print("Dictionary and corpus have been prepared")
  print("modeling data...")
  # make a model
  lda_model = gensim.models.LdaMulticore(corpus=corpus,           # vectorised corpus - list of lists of tupols          
                                     id2word=id2word,         # gensim dictionary - mapping words to IDS
                                     num_topics=15,           # number of topics
                                     random_state=100,        # set for reproducability
                                     chunksize=100,            # batch data for efficiency
                                     passes=10,               # number of full passess over data
                                     iterations=1000,          # related to document rather than corpus
                                     per_word_topics=True,    # define word distributions
                                     minimum_probability=0.1,
                                     gamma_threshold=0.005)   # the minimum step-size improvement, if not exceeds, the model stops
    
  print("Data have succesfully been modeled.")
  # Compute Perplexity
  print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

  # Compute Coherence Score
  coherence_model_lda = CoherenceModel(model=lda_model, 
                                       texts=data_processed, 
                                       dictionary=id2word, 
                                       coherence='c_v')

  coherence_lda = coherence_model_lda.get_coherence()
  print('\nCoherence Score: ', coherence_lda)
   
  print("generating and saving LDAvis plot")
  vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
   
  # save the LDA plot
  pyLDAvis.save_html(vis,os.path.join("..", "reports", "figures", "philosophy_LDAvis.html"))
  print(f"Done!, the vizualization is available at {os.path.join('..', 'reports', 'figures', 'philosophy_LDAvis.html')}")
  
if __name__ =="__main__":
  df = data_wrapping()
  texts = chunking()
  data_processed = process_word_no_grams()
  lda_modeling()

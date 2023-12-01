import pandas as pd
import string
import fr_core_news_md
import de_core_news_md
import matplotlib.pyplot as plt
import gensim
from stop_words import get_stop_words
from gensim.models import CoherenceModel
import gensim
import pyLDAvis
import pyLDAvis.gensim_models
from nltk.corpus import stopwords
"""
LDA related functions
These functions follow a certain order of process to complete the desired LDA model
"""
""" Language-dependent variables declarations """
# FR
#df = pd.read_csv("Ads1FR.csv")
nlp = fr_core_news_md.load(disable=['parser', 'ner'])
stop_words = set(get_stop_words('french')) | set(stopwords.words('french'))
# DE
#df = pd.read_csv("Ads1DE.csv")
#nlp = de_core_news_md.load(disable=['parser', 'ner'])
#stop_words = set(get_stop_words('german')) | set(stopwords.words('german'))
""" End of declaration """

def clean_text(text):
  """
  Cleans the text by removing unnecessary punctuation symbols
  """
  delete_dict = {sp_char: '' for sp_char in string.punctuation}
  delete_dict[' '] =' '
  table = str.maketrans(delete_dict)
  text1 = text.translate(table)
  textArr= text1.split()
  text2 = ' '.join([w for w in textArr if ( not w.isdigit() and
                                           ( not w.isdigit() and len(w)>3))])
  return text2.lower()

def remove_stopwords(text):
  """
  Removes stopwords from a parameter text
  """
  textArr = text.split(' ')
  rem_text = " ".join([i for i in textArr if i not in stop_words])
  return rem_text

def lemmatization(texts,allowed_postags=['NOUN', 'ADJ', 'VERB']):
	output = []
	for sent in texts:
		doc = nlp(sent)
		output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
	return output

def lda_total(df, topics_range):
  """
  Main LDA Model function, cleans text, which is expected to be a csv column,
  It then removes stopwords,
  Applies lemmatization,
  Converts the result to a document term frequency dictionary
  And finally builds an LDA model using Gensim's library
  With said model, Perplexity and coherence are calculated.
  """
  print('Cleaning text and removing stopwords')
  #Clean text
  df['Description'] = df['Description'].apply(clean_text)
  #Remove stopwords
  df['Description'] = df['Description'].apply(remove_stopwords)
  #Lemmatization
  print('Done!')
  print('\nApplying Lemmatization...')

  text_list = df['Description'].tolist()
  print('Test list: ',text_list[2])
  #nlp = de_core_news_md.load(disable=['parser', 'ner'])
  tokenized_ads = lemmatization(text_list)
  print('List[2]: ',tokenized_ads[2])
  # convert to document term frequency:
  id2word = gensim.corpora.Dictionary(tokenized_ads)
  corpus = [id2word.doc2bow(rev) for rev in tokenized_ads]
  # Creating the object for LDA model using gensim library
  LDA = gensim.models.ldamodel.LdaModel
  # Build LDA model
  print('Building LDA model...')
  lda_model = LDA(corpus=corpus, id2word=id2word,
                num_topics=topics_range, random_state=100,
                chunksize=1000, passes=100,iterations=250)
  # print lda topics with respect to each word of document
  lda_model.print_topics()
  print('Top topics:',lda_model.top_topics(corpus))
  doc_lda = lda_model[corpus]
  
  # calculate perplexity and coherence
  print('\nPerplexity: ', lda_model.log_perplexity(corpus, total_docs=10000)) 
  
  # calculate coherence
  coherence_model_lda = CoherenceModel(model=lda_model,
                                     texts=tokenized_ads, dictionary=id2word ,
                                     coherence='u_mass', processes=4)
  print('Calculating coherence...')
  coherence_lda = coherence_model_lda.get_coherence()
  print('Coherence: ', coherence_lda)

  # Visualising the Data
  pyLDAvis.enable_notebook()
  pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

'''Hello! This module of code is a part of the larger pyResearchInsights project.
This file was earlier named as Temp_Gensim_Code; code is now bifurcated into Gensim code (this) and a seperate
visualization code that will be added to the repository as well.

Checkout the Bias README.md for an overview of the project.

Sarthak J. Shetty
24/11/2018'''

'''Natural Language toolkit. Here we download the commonly used English stopwords'''
import nltk; nltk.download('stopwords')
'''Standard set of functions for reading and appending files'''
import re
'''Pandas and numpy is a dependency used by other portions of the code.'''
import numpy as np
import pandas as pd
'''Think this stands for pretty print. Prints out stuff to the terminal in a prettier way'''
from pprint import pprint
'''Importing OS to get current working directory (cwd) to tackle abstracts_log_name edge cases'''
import os
'''Contains the language model that has to be developed.'''
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from pyResearchInsights.common_functions import status_logger
from pyResearchInsights.Visualizer import visualizer_main
'''Industrial level toolkit for NLP'''
import spacy

import pyLDAvis
import pyLDAvis.gensim_models

'''Make pretty visualizations'''
import matplotlib as plt

'''Library to log any errors. Came across this in the tutorial.'''
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'com', 'https', 'url', 'link', 'abstract', 'author', 'chapter', 'springer', 'title', "the", "of", "and", "in", "to", "a", "is", "for", "from", "with", "that", "by", "are", "on", "was", "as", 
	"were", "url:", "abstract:", "abstract",  "author:", "title:", "at", "be", "an", "have", "this", "which", "study", "been", "not", "has", "its", "also", "these", "this", "can", "a", 'it', 'their', "e.g.", "those", "had", "but", "while", "will", "when", "only", "author", "title", "there", "our", "did", "as", "if", "they", "such", "than", "no", "-", "could"])

def data_reader(abstracts_log_name, status_logger_name):
	'''This wherer the file is being parsed from to the model'''
	data_reader_start_status_key = abstracts_log_name+".txt is being ported to dataframe"
	status_logger(status_logger_name, data_reader_start_status_key)

	try:
		'''If the NLP_Engine script is run independently, not as part of the pipeline as a whole, there would be no filename_CLEAND.txt.
		This ensures that that file can be processed independently.'''
		abstracts_txt_file_name = (abstracts_log_name.split(".txt")[0]) + "_" + 'CLEANED.txt'
		textual_dataframe = pd.read_csv(abstracts_txt_file_name, delimiter="\t")
	except FileNotFoundError:
		textual_dataframe = pd.read_csv(abstracts_log_name, delimiter="\t")

	data_reader_end_status_key = abstracts_log_name + ".txt has been ported to dataframe"	
	status_logger(status_logger_name, data_reader_end_status_key)

	return textual_dataframe

def textual_data_trimmer(textual_dataframe, status_logger_name):
	'''Converts each of the abstracts in the file into a list element, of size = (number of abstracts)'''
	textual_data_trimmer_start_status_key = "Trimming data and preparing list of words"
	status_logger(status_logger_name, textual_data_trimmer_start_status_key)

	textual_data = textual_dataframe.values.tolist()

	textual_data_trimmer_end_status_key = "Trimmed data and prepared list of words"
	status_logger(status_logger_name, textual_data_trimmer_end_status_key)

	return textual_data

def sent_to_words(textual_data, status_logger_name):
	'''Removing unecessary characters and removing punctuations from the corpus. Resultant words are then tokenized.'''
	sent_to_words_start_status_key = "Tokenizing words"
	status_logger(status_logger_name, sent_to_words_start_status_key)

	for sentence in textual_data:
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
	textual_data = list(sent_to_words(textual_data, status_logger_name))
	
	sent_to_words_end_status_key = "Tokenized words"
	status_logger(status_logger_name, sent_to_words_end_status_key)	

	return textual_data

def bigram_generator(textual_data, status_logger_name):
	'''Generating bigram model from the words that are in the corpus.'''
	'''Bigrams: Words that occur together with a high frequency,'''
	bigram_generator_start_status_key = "Generating word bigrams"
	status_logger(status_logger_name, bigram_generator_start_status_key)
	
	bigram = gensim.models.Phrases(textual_data, min_count=5, threshold=100)
	bigram_mod = gensim.models.phrases.Phraser(bigram)

	bigram_generator_end_status_key = "Generated word bigrams"
	status_logger(status_logger_name, bigram_generator_end_status_key)	

	return bigram_mod

def remove_stopwords(textual_data, status_logger_name):
	'''This function removes the standard set of stopwords from the corpus of abstract words.
	We've added a bunch of other words in addition.'''
	remove_stopwords_start_status_key = "Removing stopwords"
	status_logger(status_logger_name, remove_stopwords_start_status_key)
	
	return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in textual_data]
	
	remove_stopwords_end_status_key = "Removed stopwords"
	status_logger(status_logger_name, remove_stopwords_end_status_key)

def format_topics_sentences(ldamodel, corpus, texts):
	'''This function generates a dataframe that presents the dominant topic of each entry in the dataset'''
	sent_topics_df = pd.DataFrame()
	for i, row in enumerate(ldamodel[corpus]):
		row = sorted(row, key=lambda x: (x[1]), reverse=True)
		for j, (topic_num, prop_topic) in enumerate(row):
			if j == 0:
				wp = ldamodel.show_topic(topic_num)
				topic_keywords = ", ".join([word for word, prop in wp])
				sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
			else:
				break
	sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
	contents = pd.Series(texts)
	sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

	return (sent_topics_df)

def make_bigrams(textual_data, status_logger_name):
	'''Generates multiple bigrams of word pairs in phrases that commonly occuring with each other over the corpus'''
	make_bigrams_start_status_key = "Generating bigrams"
	status_logger(status_logger_name, make_bigrams_start_status_key)

	bigram_mod = bigram_generator(textual_data, status_logger_name)
	return [bigram_mod[doc] for doc in textual_data]
	
	make_bigrams_end_status_key = "Generated bigrams"
	status_logger(status_logger_name, make_bigrams_end_status_key)

def lemmatization(status_logger_name, textual_data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
	'''Reducing a word to the root word. Running  -> Run for example'''
	lemmatization_start_status_key = "Beginning lemmatization"
	status_logger(status_logger_name, lemmatization_start_status_key)

	texts_out = []
	try:
		nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
	except OSError:
		from spacy.cli import download
		download('en_core_web_sm')
		nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
	for sent in textual_data:
		doc = nlp(" ".join(sent))
		texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

	lemmatization_end_status_key = "Ending lemmatization"
	status_logger(status_logger_name, lemmatization_end_status_key)

	return texts_out

def nlp_engine_main(abstracts_log_name, status_logger_name, num_topics = None, num_keywords = None, mallet_path = None):
	nlp_engine_main_start_status_key = "Initiating the NLP Engine"
	status_logger(status_logger_name, nlp_engine_main_start_status_key)

	'''We can arrive at logs_folder_name from abstracts_log_name, instead of passing it to the NLP_Engine function each time'''
	if('Abstract' in abstracts_log_name):
		logs_folder_name = abstracts_log_name.split('Abstract')[0][:-1]
	else:
		'''If the user points to an abstracts_log_name that does not contain 'Abstract' and lies at the current working directory then set the logs_folder_name as cwd'''
		logs_folder_name = ''

	if(logs_folder_name == ''):
		'''This condition is required, if the file is located at the directory of the pyResearchInsights code.'''
		logs_folder_name = logs_folder_name + os.getcwd()

	'''Declaring the number of topics to be generated by the LDA model'''
	if num_topics == None:
		'''If the user has not provided this argument then set to 10'''
		num_topics = 10

	'''Declaring the number of keywords to be presented by the Visualizer'''
	if num_keywords == None:
		'''If the user has not provided this argument then set to 20'''
		num_keywords = 20

	'''Extracts the data from the .txt file and puts them into a Pandas dataframe buckets'''
	textual_dataframe = data_reader(abstracts_log_name, status_logger_name)
	'''Rids the symbols and special characters from the textual_data'''
	textual_data = textual_data_trimmer(textual_dataframe, status_logger_name)
	'''Removes stopwords that were earlier downloaded from the textual_data'''
	textual_data_no_stops = remove_stopwords(textual_data, status_logger_name)
	'''Prepares bigrams'''
	textual_data_words_bigrams = make_bigrams(textual_data_no_stops, status_logger_name)
	'''Lemmatization: Running -> Run'''
	textual_data_lemmatized = lemmatization(status_logger_name, textual_data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
	'''Creating a dictionary for each term as the key, and the value as their frequency in that sentence.'''
	id2word = corpora.Dictionary(textual_data_lemmatized)

	texts = textual_data_lemmatized
	'''Creating a dictionary for the entire corpus and not just individual abstracts and documents.'''
	corpus = [id2word.doc2bow(text) for text in texts]

	'''Builds the actual LDA model that will be used for the visualization and inference'''
	lda_model_generation_start_status_key = "Generating the LDA model using default parameter set"
	status_logger(status_logger_name, lda_model_generation_start_status_key)

	if(mallet_path):
		lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics = num_topics, id2word=id2word)

		'''Generating a dataset to show which '''
		df_topic_sents_keywords = format_topics_sentences(ldamodel = lda_model, corpus = corpus, texts = textual_data)
		df_dominant_topic = df_topic_sents_keywords.reset_index()
		df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
		df_dominant_topic.to_csv(logs_folder_name + '/Master_Topic_Per_Sentence.csv')

		'''Generating a dataset to present the percentage of papers under each topic, their keywords and number of papers'''
		sent_topics_sorteddf_mallet = pd.DataFrame()
		sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
		for i, grp in sent_topics_outdf_grpd:
			sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], axis=0)
		sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
		topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
		topic_contribution = round(topic_counts/topic_counts.sum(), 4)
		sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
		sent_topics_sorteddf_mallet.head()
		sent_topics_sorteddf_mallet['Number_Papers'] = [topic_counts[count] for count in range(num_topics)]
		sent_topics_sorteddf_mallet['Percentage_Papers'] = [topic_contribution[count] for count in range(0, num_topics)]
		sent_topics_sorteddf_mallet.to_csv(logs_folder_name+'/Master_Topics_Contribution.csv')

		'''Converting the mallet model to LDA for use by the Visualizer code'''
		lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model)

	else:
		lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = id2word, num_topics = num_topics, random_state = 100, update_every = 1, chunksize = 100, passes = 10, alpha = 'auto', per_word_topics = True)

	lda_model_generation_end_status_key = "Generated the LDA model using default parameter set"
	status_logger(status_logger_name, lda_model_generation_end_status_key)

	perplexity_score = lda_model.log_perplexity(corpus)

	perplexity_status_key = "Issued perplexity:"+" "+str(perplexity_score)
	status_logger(status_logger_name, perplexity_status_key)

	nlp_engine_main_end_status_key = "Idling the NLP Engine"
	status_logger(status_logger_name, nlp_engine_main_end_status_key)

	'''Importing the visualizer_main function to view the LDA Model built by the NLP_engine_main() function'''
	visualizer_main(lda_model, corpus, id2word, textual_data_lemmatized, num_topics, num_keywords, logs_folder_name, status_logger_name)

	return 0
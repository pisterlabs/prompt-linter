from django.core.management.base import BaseCommand, CommandError
from visualiser.models import *
from api.models import *

import pickle
#Based on tutorial located at:
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

#imports
import sys
import re
import numpy as np
import pandas as pd
import json
import requests
import os.path
import os

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning, module='gensim')
warnings.filterwarnings("ignore",category=UserWarning, module='numpy')
warnings.filterwarnings("ignore",category=DeprecationWarning, module='numpy')

# File reading
from docx import Document
import PyPDF2

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import models


# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
warnings.filterwarnings("ignore",category=DeprecationWarning)

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer



def run(filepath, numOfTopics):
	# function definitions

	def trim_data(data):	#trims urls and superflous data out
		data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
		data = [re.sub('\S*http\S*\s?', '', sent) for sent in data]
		data = [re.sub('\S*https\S*\s?', '', sent) for sent in data]
		data = [re.sub('\s+', ' ', sent) for sent in data]
		data = [re.sub("\'", "", sent) for sent in data]
		return data

	def sent_to_words(sentences):	#yields tokenised set
		for sentence in sentences:
			yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

	def remove_stopwords(texts):	#remove useless words for topic generation
		# set stopwords
		stop_words = stopwords.words('english')
		stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
		return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

	def make_bigrams(texts):	#consider words that may have a different meaning when used in pairs
		return [bigram_mod[doc] for doc in texts]

	def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):	#remove suffixes to package similar words together despite tense
		nlp = spacy.load('en', disable=['parser', 'ner'])
		texts_out = []
		for sent in texts:
			doc = nlp(" ".join(sent))
			texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
		return texts_out

	def count_freq(text):	# Calculate frequency distribution given a string of text
		ps = PorterStemmer()
		stopWords = set(stopwords.words("english"))
		words = word_tokenize(text)
		freqTable = dict()
		for word in words:
			word = word.lower()
			if word in stopWords:
				continue
			if word in freqTable:
				ps.stem(word)
				freqTable[word] += 1
			else:
				freqTable[word] = 1
		print(freqTable)

	def get_freq_json(freq_corpus, filename):	#outputs the word collection to database
		string_corpus = ''.join(str(e) for e in freq_corpus)
		result_words = re.findall(r"'(.*?)'", string_corpus, re.DOTALL)
		result_nums = re.findall(r"-?\d+\.?\d*", string_corpus, re.DOTALL)

		#possible ordering of frequencies here but may mess up words
		dict_wordsfreq = dict(zip(result_words, result_nums))
		word_lst = []
		for word, freq in dict_wordsfreq.items():
			word_obj = Word(word=word, frequency=freq)
			# print(tmp)
			word_lst.append(word_obj)

		# json_dict = json.dumps(dict_lst)
		words_obj = Words(words=word_lst, document = filename )
		words_obj.save()

	def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):	#Computes coherence calculation
		"""
		Compute c_v coherence for various number of topics

		Parameters:
		----------
		dictionary : Gensim dictionary
		corpus : Gensim corpus
		texts : List of input texts
		limit : Max num of topics

		Returns:
		-------
		model_list : List of LDA topic models
		coherence_values : Coherence values corresponding to the LDA model with respective number of topics
		"""
		coherence_values = []
		model_list = []

		# Mallet Path Assignment: Alter to the location of mallet on your PC according to the format below.
		# Sets the environment valriable for MALLET_HOME: Needs to be a full path to your mallet folder
		os.environ.update({'MALLET_HOME':r'C:/mallet-2.0.8'})

		# Assigns the mallet path
		# This is the path from the C: Drive without the drive lettering.
		mallet_path = '/mallet-2.0.8/bin/mallet'


		for num_topics in range(start, limit, step):
			model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
			#print(model.show_topics(formatted=False))
			model_list.append(model)
			coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
			coherence_values.append(coherencemodel.get_coherence())

		return model_list, coherence_values

	def get_topic_json(optimal_model):	#creates a file of the topics and keywords
		topic_obj_lst = []
		bone_obj_lst = []

		for item in (optimal_model.print_topics()):
			topic_name, tmp = item
			word_freq = tmp.split(" + ")

			result_words = [i.split('*')[1] for i in word_freq]
			result_nums = [i.split('*')[0] for i in word_freq]

			dict_wordsfreq = dict(zip(result_words, result_nums))
			word_lst = []
			name_lst = []

			for word, freq in dict_wordsfreq.items():

				word_obj = Word(word=word.replace('"', ''), frequency=freq)
				# print(type(word.replace('"', '')))
				bone_obj = Fishbone(name=word.replace('"',''))
				# print(tmp)
				word_lst.append(word_obj)
				name_lst.append(bone_obj)

			topic_obj_lst.append(Topic(keywords=word_lst, topic=topic_name))
			bone_obj_lst.append(Fishbone(name=str(topic_name), children=name_lst))

		topics_obj = Topics(data=topic_obj_lst, document=filename)
		fishbone_obj = Fish(name=filename, children=bone_obj_lst)
		topics_obj.save()
		fishbone_obj.save()

	def format_topics_sentences(ldamodel, corpus, texts):	#aggregates topcis into presentation table

		# Init output
		sent_topics_df = pd.DataFrame()

		# Get main topic in each document
		for i, row in enumerate(ldamodel[corpus]):
			row = sorted(row, key=lambda x: (x[1]), reverse=True)
			# Get the Dominant topic, Perc Contribution and Keywords for each document
			for j, (topic_num, prop_topic) in enumerate(row):
				if j == 0:  # => dominant topic
					wp = ldamodel.show_topic(topic_num)
					topic_keywords = ", ".join([word for word, prop in wp])
					sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
				else:
					break
		sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

		# Add original text to the end of the output
		contents = pd.Series(texts)
		sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
		return(sent_topics_df)
	try:
		# Import the file
		print("Parsing Uploaded File")
		basePath = os.path.dirname(os.path.abspath(__file__))
		dir = '/documents/' + filepath

		# Check the extension to determine how to run the LDA
		filename, file_extension = os.path.splitext(filepath)

		if file_extension == '.txt':
		    #TEXT FILE
		    data = open(basePath + dir,"rb")

		elif file_extension == '.json':
			#JSON FILE
			df = pd.read_json(basePath + dir)
			# Convert to list
			# Otherwise use this when using command to run. I.E your path to the local host
			# df = pd.read_json(basePath + "/tweets_small.json"

			# Convert to list
			raw_data = df.text.tolist()

			data = trim_data(raw_data)


		elif file_extension == '.docx':
			#dosent work
		    data = open(basePath + dir, "r", encoding="ISO-8859-1").read()

		elif file_extension == '.pdf':
		    pdf_file = open(basePath + dir,"rb")
		    read_pdf = PyPDF2.PdfFileReader(pdf_file)
		    data = []
		    for i in range(read_pdf.getNumPages()):
		        page = read_pdf.getPage(i)
		        data.append(page.extractText())

		else:
		    print("File not supported")


		print("Parsing File A Success")

	except Exception as e:
		fp = "visualiser/management/commands/documents/" + filepath
		doc = FileDoc.objects.get(document=fp)
		doc.status = 'Error'
		doc.save()
		print(e)
		raise CommandError('Error parsing file.')

	s = filepath
	filename = s[:s.find(".")]

	try:

		print("Generating Bigrams")
		# run the methods in the correct order
		data_words = list(sent_to_words(data))
		bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
		bigram_mod = gensim.models.phrases.Phraser(bigram)
		print("Bigram generated")

		print("Removing Stopwords")
		data_words_nostops = remove_stopwords(data_words)
		print("Stop words removed")

		print("Lematizing the Data")
		data_words_bigrams = make_bigrams(data_words_nostops)
		data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
		print("Data lematized")

		print("Creating the Corpus")
		id2word = corpora.Dictionary(data_lemmatized)
		texts = data_lemmatized
		corpus = [id2word.doc2bow(text) for text in texts]
		print("Corpus created")

		# print("Heres the frequency corpus")
		frequency_corpus = [[(id2word[id], freq) for id, freq in cp] for cp in corpus]
		get_freq_json(frequency_corpus, filename)

		print("This is going to take a while\n")
		# Can take a long time to run.

		print("Computing Coherence Values")
		numOfTopics  = int(numOfTopics)
		if numOfTopics == -1:
			top = 0;
			count = 0;
			model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, limit=35, start=5, step=5)
			for i in coherence_values:
				if i > top:
					topic_count = count
					top = i
				count+=1
		else:
			k = numOfTopics
			plusone = k+1
			model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, limit=plusone, start=k, step=1)
			topic_count = 0;

		# print("Heres a graph of the topic count increase vs coherence\n")
		# Show graph of results
		limit=40; start=2; step=6;
		x = range(start, limit, step)
		# plt.plot(x, coherence_values)
		# plt.xlabel("Num Topics")
		# plt.ylabel("Coherence score")
		# plt.legend(("coherence_values"), loc='best')
		# plt.show()

		"""If the coherence score seems to keep increasing,
		it may make better sense to pick the model that gave the highest CV
		before flattening out. This is exactly the case here.
		"""

		# Print the coherence scores
		# for m, cv in zip(x, coherence_values):
		# 	print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

		#then select the model where it peaked and began to flatten out
		#user input required here
		# print("There are ", len(model_list), " models")
		optimal_model = model_list[topic_count] #TODO this doesnt change
		model_topics = optimal_model.show_topics(formatted=False)
		# print("number of topics is: ")
		# print(optimal_model.num_topics)
		# print("Optimal model curve: ")
		# print("Heres the topics")
		# print("\n")
		# print("\n")
		final_topics = optimal_model.print_topics(num_words=100)
		# print(final_topics)

		pickle.dump(optimal_model, open("big_optimal_model.pkl", "wb"))

		get_topic_json(optimal_model)

	except Exception as e:
		fp = "visualiser/management/commands/documents/" + filepath
		doc = FileDoc.objects.get(document=fp)
		doc.status = 'Error'
		doc.save()
		print(e)
		raise CommandError( 'Error running LDA')

	try:
		print('Updating analysis status')
		fp = "visualiser/management/commands/documents/" + filepath
		doc = FileDoc.objects.get(document=fp)
		doc.status = 'Complete'
		doc.save()

		print('LDA Analysis Complete. View the graphs for the data through the homepage.')
	except Exception as e:
		fp = "visualiser/management/commands/documents/" + filepath
		doc = FileDoc.objects.get(document=fp)
		doc.status = 'Error'
		doc.save()
		print(e)
		raise CommandError('Error setting file status as complete')



# Bottom to classes are the entry points for the web interface to call this script and to run it in the background.
class Command(BaseCommand):
	help = 'Initialises and runs the analysis on a dataset. Requires a file path as a parameter'

	def add_arguments(self, parser):
		# This sets the number of command line arguments as 1 and sets it to the variable file
		parser.add_argument('file', nargs='+')

class Command(BaseCommand):
	help = 'Initialises and runs the analysis on a dataset. Requires a file path as a parameter'

	def add_arguments(self, parser):
		# This sets the number of command line arguments as 1 and sets it to the variable file
		parser.add_argument('file', nargs='+')

	def handle(self, *args, **kwargs):

		try:

			file = kwargs['file']
			filepath = file[0]
			numOfTopics = file[1]
			print("LDA Analysis Starting")
		except Exception as e:
			fp = "visualiser/management/commands/documents/" + filepath
			doc = FileDoc.objects.get(document=fp)
			doc.status = 'Error'
			doc.save()
			print(e)
			raise CommandError( 'Invalid Arguments used.')

		try:

			run(filepath, numOfTopics)
			sys.exit(0)


		except Exception as e:
			fp = "visualiser/management/commands/documents/" + filepath
			doc = FileDoc.objects.get(document=fp)
			doc.status = 'Error'
			doc.save()
			print(e)
			raise CommandError( 'Error running LDA. The program terminated unexpectedly. Check that the django-server is running and re-upload the document')

		# Check File Exists otherwise throw error. Authenticate file type etc.
		# sets the status to complete at teh end

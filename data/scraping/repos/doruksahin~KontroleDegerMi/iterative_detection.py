#coding=utf-8

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

import sys
import time
from gensim.models.ldamulticore import LdaMulticore

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import gensim
import time
# from pprint import pprint
import logging
import matplotlib.pyplot as plt
import json
import pickle
from nltk.stem import PorterStemmer
import snowballstemmer
import argparse
import chardet

from os import listdir, makedirs
from os.path import isfile, join






'''
criteria = "c_v"
passes = 10
max_topic = 9 # In iteration, this goes up to 8, not 9.
iteration_range = 11 # This goes up to 10, not 11.
'''


logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('word_tokenize')
nltk.download('wordnet')


def writeFile(newsFile, num_topics, num_words, model, stem=""):
	sentence_array = []
	try:
		makedirs(newsFile[:-4])
	except:
		pass
	for i in range(1, num_topics + 1):
		print(model.print_topic((i - 1), num_words))
		sentence_array.append("({},\'{}\')".format(i - 1, model.print_topic(i - 1, num_words).encode("utf-8")).replace(" ", ""))

	file = open("./{}/{}num_topics={}-num_words={}.csv".format(newsFile[:-4], stem, num_topics, num_words), "w")
	for sentence in sentence_array:
		sentence_num = sentence.replace("[", "").replace("(", "")[0]
		for word in sentence[sentence.find("\'") + 1:sentence.find("\'", sentence.find("\'") + 1)].split("+"):
			score = word[0:word.find("*")]
			keyword = word[word.find("\"") + 1:len(word) - 1]
			file.write(sentence_num + ";" + keyword + ";" + score + "\n")

	topics = []
	for sentence in sentence_array:
		topic = {}
		sentence_num = sentence.replace("[", "").replace("(", "")[0]
		topic['topic_id'] = sentence_num
		topic['keywords'] = []
		for word in sentence[sentence.find("\'") + 1:sentence.find("\'", sentence.find("\'") + 1)].split("+"):
			word_info = {}
			score = word[0:word.find("*")]
			keyword = word[word.find("\"") + 1:len(word) - 1]
			word_info['keyword'] = keyword
			word_info['score'] = score
			topic['keywords'].append(word_info)
		topics.append(topic)
	with open("./{}/{}num_topics={}-num_words={}.json".format(newsFile[:-4], stem, num_topics, num_words).replace("\n", ""), "w+") as json_file:  # overwrites.
		json.dump(topics, json_file, indent=2, ensure_ascii=False)

	arr = []
	for sentence in sentence_array:
		sentenceArr = []
		words = sentence[sentence.find("\'") + 1:sentence.find("\'", sentence.find("\'") + 1)].split("+")
		for word in words:
			keyword = word[word.find("\"") + 1:len(word) - 1]
			sentenceArr.append(keyword)
		arr.append(sentenceArr)
	file = open("./{}/{}num_topics={}-num_words={}.txt".format(newsFile[:-4], stem, num_topics, num_words), 'wb')
	pickle.dump(arr, file)



# Verilen dosyayi stopwords'e eklemek.
def updateStopwords(stopwords_file):
	with open("./" + stopwords_file, "r") as turkishStopwordsFile:
		turkish_stopwords_arr = []
		for line in turkishStopwordsFile.readlines():
			line = line.replace(" ", "_")
			line = line.replace("\n", "")
			turkish_stopwords_arr.append(line)
		stop_words.update(turkish_stopwords_arr)


def deleteStopwords():
	with open("./extracted/{}".format(newsFile), "r") as readFile:
		for line in readFile.readlines():
			emoji_pattern = re.compile("["
									   u"\U0001F600-\U0001F64F"  # emoticons
									   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
									   u"\U0001F680-\U0001F6FF"  # transport & map symbols
									   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
									   "]+", flags=re.UNICODE)
			line = line.decode("utf-8", "ignore")
			line = emoji_pattern.sub(r'', line)
			lineArr = []


			line = re.sub(r'@\S*', '', line.strip())                                # @ahmethc icin eklendi.
			# line = re.sub(r'aracılığıyla', '', line.encode("utf-8"))                # @ahmethc icin eklendi.
			# line = line.decode("utf-8")
			# line = re.sub(r'Ahmet Hakan yazdı', '', line.encode("utf-8"))           # @ahmethc icin eklendi
			# line = line.decode("utf-8")

			line = re.sub(r'https\S*', '', line.strip())                            # Linkleri cikar.
			line = re.sub(r'([0-9]*)[\.,]*([0-9]*)', '', line.strip())              # Sayilari cikar.
			line = line.encode("utf-8", "ignore")
			line = re.sub(r"[^a-zA-Z0-9@#ŞÖÇİĞÜşöçığü ]", "", line) # Emojileri cikar.
			line = line.decode("utf-8", "ignore")
			for word in line.split():
				if not word.lower() in stop_words:
					lineArr.append(word.lower())
			texts.append(lineArr)


def addBiagramTriagram():
	global texts
	bigram = gensim.models.Phrases(texts)
	texts = [bigram[line] for line in texts]
	trigram = gensim.models.Phrases(texts)
	texts = [trigram[line] for line in texts]

def deleteStopwordsBiagram():
	for i in range(len(texts)):
		len_line_arr = len(texts[i])
		j = 0
		while j < len_line_arr:
			if texts[i][j].lower() in stop_words:
				del texts[i][j]
				len_line_arr -= 1
				j -= 1
			j += 1



if __name__ == '__main__':
	texts = []

	# config.txt dosyasinin islenmesi.
	parameters_array = []
	for parameter in open("config.txt", "r").readlines():
		parameters_array.append(parameter.split(" ")[2]) # passes = 10
	criteria = parameters_array[0]
	iteration_range = int(parameters_array[1]) + 1
	passes = int(parameters_array[2])
	max_topic = int(parameters_array[3]) + 1
	num_words = int(parameters_array[4])
	stem_switch_input = parameters_array[5].replace("\n", "")
	language = parameters_array[6]


	startTime = time.time()

	if stem_switch_input == "Yes":
		stem_switch = True
	else:
		stem_switch = False



	if language == 'tr':
		stop_words = set(stopwords.words('turkish'))
		updateStopwords("turkish_stopwords.txt")
		updateStopwords("turkish_dynamic_stopwords.txt")
		ts = snowballstemmer.TurkishStemmer()

	elif language == 'en':
		stop_words = set(stopwords.words('english'))
		updateStopwords("english_stopwords.txt")
		updateStopwords("english_dynamic_stopwords.txt")
		ts = PorterStemmer()

	else:
		print("Boyle bir dil secenegi yok.")
		sys.exit(1)
		

	# ts = snowballstemmer.EnglishStemmer()



	files = [f for f in listdir("./extracted") if isfile(join("./extracted", f))]
	for newsFile in files:
		texts = []
		deleteStopwords()
		addBiagramTriagram()
		deleteStopwordsBiagram()
		 

		dictionary = Dictionary(texts)
		corpus = [dictionary.doc2bow(text) for text in texts]


		topics_array = []                       # Her topic'in en iyi iterasyonunun modelini iceriyor.
		for num_topics in range(1, max_topic):
			print("Num Topics: {}".format(num_topics))
			modelArray = []                     # Bir topic sayisi icin tum modeller.
			highest = (0, 0)                    # (iteration_count, coherence_score)
			for iterations in range(1,iteration_range):
				# model = LdaModel(corpus=corpus, id2word=dictionary, passes=passes, iterations=iterations, num_topics=num_topics)
				model = LdaMulticore(corpus=corpus, id2word=dictionary, passes=passes, iterations=iterations, num_topics=num_topics, workers=20)
				modelArray.append(model)
				cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
				if highest[1] < cm.get_coherence():
					highest = (iterations, cm.get_coherence())
				print("{}: {}".format(iterations, cm.get_coherence()))
			print(highest[0])
			print("")
			topics_array.append(modelArray[highest[0] - 1])
			writeFile(newsFile, num_topics, num_words, modelArray[highest[0] - 1])

		coherence_values = []               # Plotting to see better values.
		for i in range(1,max_topic):
			cm = CoherenceModel(model=(topics_array[i - 1]), texts=texts, dictionary=dictionary, coherence='c_v')
			coherence_values.append(cm.get_coherence())














		if stem_switch:
			texts = []
			deleteStopwords()
			# texts = [[word.lower() for word in re.sub("[^a-zA-Z0-9@#]", " ", line.strip()).split() if not word.lower() in stop_words] for line in open("./extracted/{}".format(newsFile), 'r').readlines()]
			if language == "tr":
				texts = [[ts.stemWord(word) for word in line if ts.stemWord(word) is not None] for line in texts]
			elif language == "en":
				texts = [[ts.stem(word) for word in line if ts.stem(word) is not None] for line in texts]
			else:
				print("No such language.")
				sys.exit(1)



			addBiagramTriagram()
			deleteStopwordsBiagram()
			'''
			bigram = gensim.models.Phrases(texts)
			texts = [bigram[line] for line in texts]
			trigram = gensim.models.Phrases(texts)
			texts = [trigram[line] for line in texts]
			'''

		
			dictionary = Dictionary(texts)
			corpus = [dictionary.doc2bow(text) for text in texts]

			topics_array_stemmed = []                       # Her topic'in en iyi iterasyonunun modelini iceriyor.
			for num_topics in range(1, max_topic):
				print(num_topics)
				print("Num Topics: {}".format(num_topics))
				modelArray = []                     # Bir topic sayisi icin tum modeller.
				highest = (0, 0)                    # (iteration_count, coherence_score)
				for iterations in range(1,iteration_range):
					# model = LdaModel(corpus=corpus, id2word=dictionary, passes=passes, iterations=iterations, num_topics=num_topics)
					model = LdaMulticore(corpus=corpus, id2word=dictionary, passes=passes, iterations=iterations, num_topics=num_topics, workers=20)
					modelArray.append(model)
					cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
					if highest[1] < cm.get_coherence():
						highest = (iterations, cm.get_coherence())
					print("{}: {}".format(iterations, cm.get_coherence()))
				print(highest[0])
				print("")
				topics_array_stemmed.append(modelArray[highest[0] - 1])
				writeFile(newsFile, num_topics, num_words, modelArray[highest[0] - 1], "stem-")

			coherence_values_stemmed = []
			for i in range(1, max_topic):
				cm = CoherenceModel(model=(topics_array_stemmed[i - 1]), texts=texts, dictionary=dictionary, coherence='c_v')
				coherence_values_stemmed.append(cm.get_coherence())



		x = range(1,max_topic)
		plt.plot(x, coherence_values, "Red")
		if stem_switch:
			plt.plot(x, coherence_values_stemmed, "Blue")
		plt.xlabel("Topic Counts")
		plt.ylabel("Coherence score")
		plt.legend(("coherence_values"), loc='best')
		plt.savefig("./{}/coherence.png".format(newsFile[:-4]))

	print("Gecen zaman: {}".format(time.time() - startTime))


import argparse
import os
import string
import sys

from os import path
from wordcloud import WordCloud

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint

def main(args):
	USER = args.username
	DIRPATH = os.path.join(os.getcwd(), USER)
	DATAPATH = os.path.join(os.getcwd(), USER, USER + ".csv")

	if not os.path.isdir(USER) or not os.path.exists(DATAPATH ):
		print("Could not find csv output of scrape.py at ", DATAPATH)
		exit()
	
	# Read data into dataframe
	data = pd.read_csv(DATAPATH)
	data.columns = ['post', 'post_type', 'caption', 'likes', 'views', 'date']

	# Save plot of likes over time
	likes_path = os.path.join(DIRPATH, "likes.png")
	plot(data, 'post', ['likes'], ['Likes'], save_path=likes_path, xlabel='Post', ylabel='Likes', title='@' + USER)

	# Generate wordcloud visualization
	data['caption_processed'] = data['caption'].map(lambda x: process_raw_caption(x))
	word_bag = ','.join(list(data['caption_processed'].values))
	wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
	wordcloud.generate(word_bag)
	WORD_CLOUD_PATH = os.path.join(USER, 'wordcloud.png')
	wordcloud.to_file(WORD_CLOUD_PATH)

	# LDA topic clustering
	texts = [[text for text in doc.split()] for doc in data['caption_processed']]
	id2word = corpora.Dictionary(texts)
	corpus = [id2word.doc2bow(text) for text in texts]
	COHERENCE_PATH = os.path.join(DIRPATH, "coherence.png")
	lda_model = get_best_lda(texts, id2word, corpus, save_path=COHERENCE_PATH, min=2, max=25)
	vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
	VIS_PATH = os.path.join(DIRPATH, "lda.html")
	pyLDAvis.save_html(vis, VIS_PATH)

	# Cluster topics from LDA results
	topic = []
	for doc in corpus:
		topic.append(max(lda_model[doc], key=lambda item: item[1])[0])
	data['topic'] = topic
	topic_plot = data.groupby('topic').size().plot(kind='pie', autopct='%.2f', title='Topics', ylabel="Nice")
	TOPIC_CLUSTER_PATH = os.path.join(DIRPATH, 'topics.png')
	topic_plot.figure.savefig(TOPIC_CLUSTER_PATH)
	

# Plots dataframe data
# TO DO: generalize this better
def plot(df, x, y, labels, save_path=None, xlabel=None, ylabel=None, title=None):
	for i, label in zip(y, labels):
		plt.plot(df[x], df[i], label=label)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.title(title)
	if save_path is not None:
		plt.savefig(save_path)
	plt.figure()

# Basic text processing:
# 1. Convert to lowercase
# 2. Remove HTML tags
# 3. Remove emojis (and other non-english symbols)
# 4. Remove punctuation
# 5. Remove spacing
# 6. Remove stopwords
def process_raw_caption(caption):
	stop_words = set(stopwords.words('english'))
	# extend stopwords if desired
	caption = caption.lower().strip()
	caption = re.compile('<.*?>').sub('', caption)
	caption = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
                      "]+", re.UNICODE).sub('', caption)
	caption = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', caption)
	caption = re.sub('\s+', ' ', caption)
	filtered_words  = [word for word in caption.split() if word not in stop_words]
	caption = ' '.join(filtered_words)
	return caption

# Select the optimal number of topics to use
# Impose a penalty scalar to prevent overfitting (want to use number of topics that 
# results in the coherence score peaking the earliest)
def get_best_lda(texts, id2word, corpus, save_path=None, min=2, max=25, penalty_scalar=1.005):
	num_topics = np.arange(min, max+1)
	coherence = []
	score_to_beat = 0
	best_n = 0
	best_score = 0
	best_lda_model = None
	for n in num_topics:
		score_to_beat *= penalty_scalar
		print("Generating LDA model with", n, "topics")
		lda_model = gensim.models.LdaMulticore(corpus=corpus,
												id2word=id2word,
												num_topics=n)
		coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
		score = coherence_model_lda.get_coherence()
		print("Coherence:", score)
		if score > score_to_beat:
			best_lda_model = lda_model
			best_n = n
			best_score = score
			score_to_beat = score
		coherence.append(score)
	print("Best coherence score is", best_score, "with", best_n, "topics")
	pprint(best_lda_model.print_topics())
	if save_path is not None:
		plt.plot(num_topics, coherence, label="Coherence")
		plt.xlabel("Num Topics")
		plt.ylabel("Coherence score")
		plt.title("LDA Number of Topics Selection")
		plt.savefig(save_path)
		plt.figure()
	return best_lda_model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("username", help="Instagram username to process. Expects to find output of scrape.py (<username>.csv) at ./<username>")
	args = parser.parse_args()
	main(args)


from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
from gensim.models.coherencemodel import CoherenceModel

from Components import Comment, Post, POSPost, POSComment
from helper import *

class MyDictionary:
	def __init__(self, posPosts, title_weight=1, use_comments=False):
		self.title_weight = title_weight
		self.use_comments = use_comments
		self.words_list = self.get_words_list(posPosts, self.title_weight, self.use_comments)
		self.dictionary = Dictionary(self.words_list)
		self.doc2bows = [self.dictionary.doc2bow(words) for words in self.words_list]
		print("- Created MyDictionary of {} words".format(len(self.dictionary)))

	def get_words_list(self, posPosts, title_weight, use_comments):
		words_list = []
		for posPost in posPosts:
			words = [pair[0] for pair in posPost.body_pos_filtered]
			title_words = [pair[0] for pair in posPost.title_pos_filtered]
			for i in range(title_weight):
				words += title_words
			if use_comments:
				for comment in posPost.comments:
					words += [pair[0] for pair in comment.text_pos_filtered]
					for reply in comment.comments:
						words += [pair[0] for pair in reply.text_pos_filtered]
			words_list.append(words)
		return words_list

class MyTfidf:
	def __init__(self, myDictionary, smartirs='ntc'):
		self.myDictionary = myDictionary
		self.model = TfidfModel(self.myDictionary.doc2bows, smartirs=smartirs)
		self.vectors = None
		self.clustering = None
		self.cluster2ids = None
		self.id2cluster = None
		print("- Created MyTfidf (smartirs={})".format(smartirs))

	def get_doc2bow(self, index):
		words = self.myDictionary.doc2bows[index]
		return self.model[words]

	def get_vector(self, index):
		if self.vectors:
			return self.vectors[index]

		doc2bow = self.get_doc2bow(index)
		vector = np.zeros(len(self.myDictionary.dictionary))
		for i, v in doc2bow:
			vector[i] = v
		return vector

	def get_vectors(self):
		if self.vectors:
			return self.vectors

		vectors = []
		for i in range(len(self.myDictionary.doc2bows)):
			vectors.append(self.get_vector(i))
		return vectors

	def cluster(self, n_clusters, add_date=False, v1=None, min_value=0):
		if self.clustering and self.cluster2ids and self.id2cluster \
			and len(self.clustering.cluster_centers_) == n_clusters:
			return
		
		vectors = self.get_vectors()

		if add_date:
			assert v1 is not None, "v1 must not be None!"
			assert len(v1) == len(vectors), "Lengths do not match!"
			for i, vector in enumerate(vectors):
				date_normalized = normalize_date(v1[i].date, min_value=min_value)
				np.append(vector, [date_normalized])

		self.clustering = MiniBatchKMeans(n_clusters=n_clusters)
		self.clustering.fit(vectors)

		self.cluster2ids = defaultdict(list)
		self.id2cluster = dict()
		for post_index, label in enumerate(self.clustering.labels_):
			self.cluster2ids[label].append(post_index)
			self.id2cluster[post_index] = label

		print("- Clustered with {} clusters, with{} date".format(n_clusters, "" if add_date else "out"))

	def get_cluster_keywords(self, cluster):
		assert self.clustering and self.cluster2ids and self.id2cluster, "Must cluster before!"
		assert cluster in self.cluster2ids, "Not a valid cluster number!"
		
		cluster_vector = np.zeros(len(self.myDictionary.dictionary))
		for post_index in self.cluster2ids[cluster]:
			cluster_vector += self.get_vector(post_index)

		keywords = [(self.myDictionary.dictionary[i], v) for i, v in enumerate(cluster_vector)]
		keywords.sort(key=lambda p: p[1], reverse=True)
		return keywords

class MyLda:
	def __init__(self, myDictionary, num_topics=100, topic_threshold=0.15):
		self.num_topics = num_topics
		self.topic_threshold = topic_threshold
		self.myDictionary = myDictionary
		self.model = LdaModel(self.myDictionary.doc2bows, \
			id2word=self.myDictionary.dictionary, \
			num_topics=num_topics)
		self.topic2ids, self.id2topics = self.get_mappings()
		self.coherenceModel = None
		print("- Created MyLda with {} topics".format(self.num_topics))

	def get_mappings(self):
		topic2ids, id2topics = defaultdict(list), defaultdict(list)
		for i, doc2bow in enumerate(self.myDictionary.doc2bows):
			topic_pairs = self.model.get_document_topics(doc2bow)
			for j, (topic, prob) in enumerate(topic_pairs):
				if prob >= self.topic_threshold or j == 0:
					topic2ids[topic].append(i)
					id2topics[i].append(topic)
		return topic2ids, id2topics

	def get_topic_terms(self, topic):
		terms = self.model.get_topic_terms(topic)
		return terms

	def get_top_topic(self):
		top_topics = self.model.top_topics(corpus=self.myDictionary.doc2bows)
		average = sum([t[1] for t in top_topics]) / self.num_topics
		return top_topics, average

	def get_perplexity(self):
		return self.model.log_perplexity(self.myDictionary.doc2bows)

	def get_coherence(self):
		if not self.coherenceModel:
			self.coherenceModel = CoherenceModel(model=self.model, \
				corpus=self.myDictionary.doc2bows, \
				dictionary=self.myDictionary.dictionary, \
				coherence='u_mass')
		return self.coherenceModel.get_coherence()

import numpy as np
import pandas as pd
from model.validation import CoherenceScore, TopicDifferenceScore, MinMaxScore, InternalExternalScore, TopicSilhouetteScore
from webconfig import config

# --------------------------------------------------------------

measure_names = { "coherence" : "Topic Coherence Score",
	"difference" : "Topic Difference Score",
	"minmax" : "Min-Max Score",
	"intext" : "Internal-External Score",
	"silhouette" : "Topic Silhouette Score" }

measure_short_names = { "coherence" : "Coherence",
	"difference" : "Difference",
	"minmax" : "Min-Max",
	"intext" : "Int-Ext",
	"silhouette" : "Silhouette" }
	
# --------------------------------------------------------------

def get_measure(measure_id, embed):
	""" Return a validation measure with the specified ID, and based on the specified
	word embedding. """
	if measure_id == "coherence":
		return CoherenceScore(embed)
	elif measure_id == "difference":
		return TopicDifferenceScore(embed)
	elif measure_id == "minmax":
		return MinMaxScore(embed)
	elif measure_id == "intext":
		return InternalExternalScore(embed)
	elif measure_id == "silhouette":
		return TopicSilhouetteScore(embed) 
	raise Exception("Unknown validation measure: %s" % measure_id)

def get_measures(measure_ids, embed):
	""" Return all validation measures with the specified IDs. """
	measures = {}
	for measure_id in measure_ids:
		measures[measure_id] = get_measure(measure_id, embed)
	return measures

# --------------------------------------------------------------

class TopicValidator:
	""" Class for generating various validation results for topics in a single topic model. """

	def get_validation_df(self, meta, embed):
		""" Get a Data Frame containing validation scores for the individual topics in
		the specified topic model. """
		if embed is None:
			return pd.DataFrame([])
		descriptors = meta.get_descriptors()
		if descriptors is None:
			return pd.DataFrame([])
		measures = get_measures(measure_names.keys(), embed)
		rows = []
		num_fmt = "%02d" if len(descriptors) < 100 else "%03d"
		for i in range(meta["k"]):
			rows.append({ "Topic" : num_fmt % (i+1), "Descriptor" : ", ".join(descriptors[i]) })
		for measure_id in measures:
			scores = measures[measure_id].evaluate_topics(descriptors)
			for i, score in enumerate(scores):
				# TODO: move rounding elsewhere?
				rows[i][measure_id] = round(score, config.get("precision", 3))
		return pd.DataFrame(rows)

	def get_topiclevel_silhouette_df(self, meta, embed):
		if embed is None:
			return pd.DataFrame([])
		descriptors = meta.get_descriptors()
		if descriptors is None:
			return pd.DataFrame([])
		measure = TopicSilhouetteScore(embed)
		topic_scores = measure.evaluate_topics(descriptors)
		rows = []
		num_fmt = "Topic %02d" if len(descriptors) < 100 else "Topic %03d"
		for i in range(meta["k"]):
			row = { "Label" : num_fmt % (i+1), "Descriptor" : ", ".join(descriptors[i]), "Number" : (i+1) }
			row["Score"] = topic_scores[i]
			rows.append(row)
		return pd.DataFrame(rows).set_index("Label")

	def get_termlevel_silhouette_scores(self, meta, embed):
		if embed is None:
			return pd.DataFrame([])
		descriptors = meta.get_descriptors()
		if descriptors is None:
			return pd.DataFrame([])
		measure = TopicSilhouetteScore(embed)
		measure.evaluate_topics(descriptors)
		return measure.topic_term_scores

	def get_topic_distance_matrix(self, meta, embed):
		if embed is None:
			return None
		descriptors = meta.get_descriptors()
		if descriptors is None:
			return None
		measure = TopicDifferenceScore(embed)
		k = len(descriptors)
		D = np.zeros((k,k))
		for i in range(k):
			# note, we populate the diagonal too
			for j in range(i, k):
				D[i,j] = measure.evaluate_distance(descriptors[i], descriptors[j])
				D[j,i] = D[i,j]
		return D

	def get_topic_similarity_matrix(self, meta, embed):
		""" Return a pairwise similarity matrix for pairs of topics, based on the currently loaded 
		model and embedding. """
		D = self.get_topic_distance_matrix(meta, embed)
		if D is None:
			return None
		return 1.0 - D

	def __get_term_similarities(self, meta, embed):
		""" Return a matrix containing the similarity between all terms appearing in topic descriptors
		in this model, which also appear in the current embedding vocabulary."""
		descriptors = meta.get_descriptors()
		all_terms = meta.get_all_descriptor_terms()
		# filter based on the embedding
		filtered_terms = []
		for term in all_terms:
			if term in embed:
				filtered_terms.append(term)
		# build the matrix
		m = len(filtered_terms)
		S = np.zeros((m, m))
		for i in range(0,m):
			for j in range(i+1,m):
				S[i, j] = embed.similarity(filtered_terms[i], filtered_terms[j])
				S[j, i] = S[i, j]
		return S, filtered_terms

	def get_term_distance_df(self, meta, embed):
		""" Returns a Data Frame of the distance between all terms appearing in topic descriptors
		in this model."""
		if embed is None:
			return None
		descriptors = meta.get_descriptors()
		if descriptors is None:
			return None
		# build the similarity matrix of terms which appear in the embedding
		S, filtered_terms = self.__get_term_similarities(meta, embed)
		D = 1.0 - S
		return pd.DataFrame(D, index=filtered_terms)

	def get_topic_pair_similarity_df(self, meta, embed, unique_only = False):
		""" Construct a Data Frame containing similarities for pairs of topics, where
		each row corresponds to a pair. """
		if embed is None:
			return None
		descriptors = meta.get_descriptors()
		if descriptors is None:
			return None
		k = len(descriptors)
		S = self.get_topic_similarity_matrix(meta, embed)
		rows = []
		num_fmt = "Topic %02d" if len(descriptors) < 100 else "Topic %03d"
		for i in range(k):
			label1 = num_fmt % (i+1)
			for j in range(i, k):
				# if we want unique pairs, skip self-similarity
				if unique_only and (i == j):
					continue
				label2 = num_fmt % (j+1)
				sim = S[i, j]
				rows.append({"topic1" : label1, "topic2" : label2, "sim" : sim})
				if not unique_only:
					rows.append({"topic1" : label2, "topic2" : label1, "sim" : sim})
		return pd.DataFrame(rows)

	def get_term_pair_similarity_df(self, meta, embed, unique_only = False):
		""" Construct a Data Frame containing similarities for pairs of terms, where
		each row corresponds to a pair. """
		if embed is None:
			return None
		descriptors = meta.get_descriptors()
		if descriptors is None:
			return None
		S, filtered_terms = self.__get_term_similarities(meta, embed)
		# determine whether each pair is intra-topic or inter-topic
		term_assignments = {}
		for i, descriptor in enumerate(descriptors):
			for term in descriptor:
				if not term in term_assignments:
					term_assignments[term] = set()
				term_assignments[term].add(i)
		# construct the Data Frame
		rows = []
		m = len(filtered_terms)
		for i in range(m):
			for j in range(i, m):
				# if we want unique pairs, skip self-similarity
				if i == j:
					if unique_only:
						continue
					sim = 1.0
				else:
					# threshold below zeros
					sim = max(S[i,j], 0)
				is_intra = len(term_assignments[filtered_terms[i]].intersection(term_assignments[filtered_terms[j]])) > 0
				rows.append({"term1" : filtered_terms[i], "term2" : filtered_terms[j], "sim" : sim, "intra" : is_intra})
				if not unique_only:
					rows.append({"term1" : filtered_terms[j], "term2" : filtered_terms[i], "sim" : sim, "intra" : is_intra})
		return pd.DataFrame(rows)	

# --------------------------------------------------------------

class ModelValidator:
	""" Class for generating various validation results across multiple topics models. """

	def get_validation_df(self, all_meta, embed):
		if embed is None:
			return None
		measures = get_measures(measure_names.keys(), embed)
		rows = []
		for meta in all_meta:
			descriptors = meta.get_descriptors()
			if descriptors is None:
				continue
			row = { "Name" : meta["id"], "Corpus" : meta["corpus"], "Topics" : len(descriptors) }
			for measure_id in measures:
				score = measures[measure_id].evaluate_model(descriptors)
				# TODO: move rounding elsewhere?
				row[measure_id] = round(score, config.get("precision", 3))
			rows.append(row)
		return pd.DataFrame(rows)

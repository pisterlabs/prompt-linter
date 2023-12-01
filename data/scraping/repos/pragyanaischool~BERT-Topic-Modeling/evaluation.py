from gensim.models.coherencemodel import CoherenceModel
from visualization import save_topic_visualization, save_topic_wordclouds
from sklearn.metrics import silhouette_score
from shared.utils import load_from_pickle
from shared.utils import dump_to_pickle
from shared.utils import dump_to_json
from shared.utils import dump_to_txt
from shared.utils import make_dirs
from collections import Counter 
from tqdm import tqdm
import pandas as pd
import numpy as np
from tm import TM
import logging
import time
import glob
import os

pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)

class Evaluation(object):
    """ Class for generating topic model and evaluation files.
    """
    def __init__(self, lang_code, method="BERT", version="1.1", norm_type="LEMMA", pre_trained_name="distilbert-base-nli-mean-tokens", 
                 num_words=10, n_neighbors=15, n_components=5, min_dist=0.0, umap_metric='cosine', min_cluster_size=30, cluster_metric='euclidean',
                 cluster_selection_method='eom', prediction_data=True, random_state=42, num_wordcloud_words=150):
                 
        self.lang_code = lang_code
        self.method = method
        self.version = version
        self.norm_type = norm_type
        self.num_words = num_words
        self.pre_trained_name = pre_trained_name
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.umap_metric = umap_metric
        self.min_cluster_size = min_cluster_size
        self.cluster_metric = cluster_metric
        self.cluster_selection_method = cluster_selection_method
        self.prediction_data = prediction_data
        self.random_state = random_state
        self.num_wordcloud_words = num_wordcloud_words
       
    def create_model(self, token_lists, output_path):
        """ Generate model & evaluation file to a given output path 
        
        :param token_lists: list of document tokens
        :param output_path: path to save model, dictionary, corpus, evaluation files
        """
        try:

            # create topic model & fit token_lists
            tm = TM(
                method=self.method, random_state=self.random_state, pre_trained_name=self.pre_trained_name, 
                n_neighbors=self.n_neighbors, n_components=self.n_components, min_dist=self.min_dist, 
                umap_metric=self.umap_metric, min_cluster_size=self.min_cluster_size, cluster_metric=self.cluster_metric, 
                cluster_selection_method=self.cluster_selection_method, prediction_data=self.prediction_data
            )

            tm.fit(token_lists)

            # define output path
            subdir = "{}_k_{}_{}_{}".format(self.lang_code, str(tm.k), self.method, self.version)
            models_path = output_path + "/models/" + subdir
            dict_path = output_path + "/dictionary/"
            corpus_path = output_path + "/corpus/"
            eval_path = output_path + "/evaluation/" + subdir
            
            # create directories
            make_dirs(output_path)
            make_dirs(models_path)
            make_dirs(dict_path)
            make_dirs(corpus_path)
            make_dirs(eval_path)
            make_dirs(eval_path + "/wordcloud")

            # extract topics 
            topics = self.extract_topics(tm.tf_idf, tm.feature_names, tm.labels)
            topic_words = self.get_topic_words(topics)
            
            # evaluate topics 
            c_v = self.compute_coherence(topic_words, tm.dictionary, tm.corpus, token_lists, measure='c_v')
            u_mass = self.compute_coherence(topic_words, tm.dictionary, tm.corpus, token_lists, measure='u_mass')
            silhouette = self.compute_silhouette(tm.umap_embeddings, tm.labels)
           
            # hyperparameters & evaluation metrics
            metrics = {
                "k"                         : tm.k,
                "lang_code"                 : self.lang_code,
                "num_docs"                  : len(token_lists),
                "method"                    : self.method,
                "version"                   : self.version,
                "norm_type"                 : self.norm_type,
                "pre_trained_name"          : self.pre_trained_name,
                "random_state"              : self.random_state,
                "n_neighbors"               : self.n_neighbors,
                "n_components"              : self.n_components,
                "min_dist"                  : self.min_dist,
                "umap_metric"               : self.umap_metric,
                "min_cluster_size"          : self.min_cluster_size,
                "cluster_metric"            : self.cluster_metric,
                "cluster_selection_method"  : self.cluster_selection_method,                
                "c_v"                       : c_v,
                "u_mass"                    : u_mass,
                "silhouette"                : silhouette
            }
            
            # save topic model, dictionary, corpus 
            tm.dictionary.save(dict_path + "/dict.gensim")
            dump_to_pickle(tm.corpus, corpus_path + "/corpus.pkl")
            dump_to_pickle(tm.sentences, eval_path + "/sentences.pkl")
            dump_to_pickle(tm.embeddings, eval_path + "/embeddings.pkl")
            dump_to_pickle(topics, models_path + "/topics.pkl")
            dump_to_pickle(tm.cluster_model, models_path + "/cluster.pkl")
            dump_to_pickle(tm.umap_model, models_path + "/umap.pkl")

            # save topic terms
            self.save_topic_terms(topics, eval_path + '/topic_terms.txt')
            
            # save metrics, topic_terms dataframe 
            dump_to_json(metrics, eval_path + "/evaluation.json", sort_keys=False)

            # save topic visualization, topic wordclouds
            save_topic_visualization(tm.embeddings, tm.labels, eval_path + "/topics.png")
            save_topic_wordclouds(topics, self.num_wordcloud_words, eval_path + "/wordcloud")

        except Exception:
            logging.error('error occured', exc_info=True)

    def compute_coherence(self, topics, dictionary, corpus, token_lists, measure):
        """ Compute coherence score for a given topic model 
        
        :param topics: topics
        :param dictionary: generated dictionary
        :param corpus: generated bow corpus
        :param token_lists: list of document tokens
        :param measure: coherence score
        """
        coherence = 0.0
        cm = CoherenceModel(
            topics=topics, 
            dictionary=dictionary, 
            corpus=corpus, 
            texts=token_lists, 
            coherence=measure
        )
        coherence = cm.get_coherence()
        coherence = "{0:.3f}".format(coherence)
        return coherence

    def compute_silhouette(self, umap_embeddings, labels):
        """ Compute silhouette score for cluster distance and relevance

        :param umap_embeddings: 
        """
        silhouette = silhouette_score(umap_embeddings, labels)
        silhouette = float("{0:.3f}".format(silhouette))
        return silhouette
    
    def extract_topics(self, tf_idf, feature_names, labels):
        """ Extract topic words and their c-TF-IDF score within each cluster
        
        :param tf_idf: tf_idf scores
        :param feature_names: dictionary words
        :param labels: cluster labels
        :return: dictionary (topic: top_n_words)
        Sample:
            {
                114 : [
                        ("cough", 0.20982713670636632)
                        ("fever", 0.0538832539146329)
                        ("throat", 0.02416355064071378)
                        ("case", 0.023852171240297093)
                        ("fever_cough", 0.02382326255347972)
                        ("illness", 0.023609866349192463)
                        ("radiograph", 0.022765546069615183)
                        ("syndrome", 0.02179170899590873)
                        ("pneumonia", 0.02169239788185851)
                ]
            }
        """
        labels = sorted(list(set(labels)))
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[::-1]

        topics = dict()
        for i, label in enumerate(labels):
            top_n_words = []
            for j in indices[i]:
                word = feature_names[j]
                c_tf_idf = tf_idf_transposed[i][j]
                top_n_words.append((word, c_tf_idf))
            top_n_words = top_n_words[::-1]
            topics[label] = top_n_words
        return topics

    def get_topic_words(self, topics):
        """ Get list of topic words from topics dictionary
        
        :param topics: dictionary (topic:top_n_words)
        :return: list of list of topic words
        Sample:
            [
                [
                    "risk", "increase", "treatment", "case", "system"
                ],
                [
                    "obesity", "weight", "surgery", "complication", "procedure"
                ],
                [
                    "tourism", "destination", "arrival", "paper", "industry"
                ]
            
            ]
        """
        topic_words = []
        for topic, top_n_words in topics.items():
            words = [word for word, c_tf_idf in top_n_words]
            topic_words.append(words)
        return topic_words

    def get_topic_terms_df(self, topics):
        """ Generate topic_term dataframe for a given model 
        
        :param topics: dictionary (topic:top_n_words)
        :return: dataframe
        """
        labels = list(topics.keys())
        
        topic_terms = []
        for topic, top_n_words in topics.items():
            top_n_words = sorted(top_n_words, key=lambda x: x[1], reverse=True)[:self.num_words]
            terms = [term for term, c_tf_idf in top_n_words]
            terms = ", ".join(terms)
            topic_terms.append(terms)

        topic_terms_df = pd.DataFrame()
        topic_terms_df['id'] = labels
        topic_terms_df['Topic terms'] = topic_terms
        return topic_terms_df

    def save_topic_terms(self, topics, output_path):
        """ Generate and save topic_terms dataframe to a given output_path
        
        :param topics: dictionary (topic:top_n_words)
        :param output_path: output path
        """
        topic_terms_df = self.get_topic_terms_df(topics)
        topic_terms_df.to_string(output_path, index=False)

 
if __name__ == "__main__":

    chunks_prep_path = abs_path = os.path.dirname(os.path.abspath("__file__")) + "/chunks_prep"
    output_path = os.path.dirname(os.path.abspath("__file__")) + "/output"	
    
    token_lists = []	
    for filename in os.listdir(chunks_prep_path):	
        chunk_no = filename.split("_")[1]	
        token_list = load_from_pickle(chunks_prep_path + "/" + filename + "/token_lists.pkl")	
        token_lists.append(token_list)	

    token_lists = [item for sublist in token_lists for item in sublist]	
    print("NUM DOCS: ", len(token_lists))	

    start_time = time.time()	
    ev = Evaluation(lang_code="en", method="BERT", version="1.1", num_words=15)	
    ev.create_model(token_lists, output_path=output_path)	
    print("--- BERT: %s seconds ---" % (time.time() - start_time))
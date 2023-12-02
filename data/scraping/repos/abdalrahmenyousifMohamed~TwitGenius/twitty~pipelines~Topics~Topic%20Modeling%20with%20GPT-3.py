import numpy as np
import pandas as pd
import openai , os
import logging
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(filename='twitter_cluster_analyzer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TwitterClusterAnalyzer:
    def __init__(self, openai_api_key):
        self.text_embeddings = None
        openai.api_key = openai_api_key
        self.model_directory = "sentence_transformer_checkpoint"


    def load_data(self, file_path):
        self.df_gender_filtered = pd.read_csv(file_path)
        self.df_gender_filtered = self.df_gender_filtered.dropna(subset=['description']).reset_index(drop=True)
        print(self.df_gender_filtered.isnull().sum())
        logging.info(f'Data loaded from {file_path}')

    def encode_text_embeddings(self):
        if os.path.exists(self.model_directory):
            logging.info('load embeddings encoded')
            embed_model = SentenceTransformer(self.model_directory)
            self.text_embeddings = embed_model.encode(self.df_gender_filtered['combined_text'].tolist(), show_progress_bar=True)
            self.df_gender_filtered['embeddings_com'] = list(self.text_embeddings)
        else:
            logging.info('create Text embeddings encoded')
            embed_model = SentenceTransformer("BAAI/bge-small-en")
            self.text_embeddings = embed_model.encode(self.df_gender_filtered['combined_text'].tolist(), show_progress_bar=True)
            self.df_gender_filtered['embeddings_com'] = list(self.text_embeddings)
            embed_model.save(self.model_directory)
        

    def cluster_tweets(self):
        umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine')
        reduced_text_embeddings = umap_model.fit_transform(self.text_embeddings)
        hdbscan_model = HDBSCAN(
            min_cluster_size=40,
            metric='euclidean',
            min_samples=5,
            prediction_data=False
        )

        text_cluster = hdbscan_model.fit(reduced_text_embeddings)
        self.df_gender_filtered['cluster'] = text_cluster.labels_
        logging.info('Tweets clustered')

    def generate_topic_titles(self, top_n=10, diversity=0.5):
        cluster_dict = {}

        for cluster, df in self.df_gender_filtered.groupby('cluster'):
            if cluster == -1:
                continue

    # find the most representative documents
        candidate_d = cosine_similarity(self.df_gender_filtered['embeddings_com'].tolist(), self.df_gender_filtered['embeddings_com'].tolist())
        candidate_d_sum = candidate_d.sum(axis=1)
        doc_list = [np.argmax(candidate_d.sum(axis=1))]
        candidates_idx = [i for i in range(len(self.df_gender_filtered)) if i != doc_list[0]]

        # filter based on maximal marginal relevance
        for _ in range(top_n - 1):
            candidate_similarities = candidate_d.sum(axis=1)[candidates_idx]
            target_similarities = np.max(candidate_d[candidates_idx][:, doc_list], axis=1)

            # Calculate MMR
            mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities
            # Update keywords & candidates
            mmr_idx = candidates_idx[np.argmax(mmr)]
            doc_list.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        cluster_dict[cluster] = {'doc': [self.df_gender_filtered['combined_text'].tolist()[idx] for idx in doc_list]}

        return cluster_dict

    def generate_topic_titles_with_prompt(self, cluster_dict):
        def get_response(prompt, temperature=0):
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message["content"]

        for i in cluster_dict.keys():
            # prompt
            documents = "\n".join([f"{c + 1}.{text}" for c, text in enumerate(cluster_dict[i]['doc'])])
            prompt = f"Generate a concise and informative topic title that captures the essence of the following collection of user BIO descriptions from the same Twitter cluster:\n{documents}"
            # response from openai
            llm_output = get_response(prompt)
            cluster_dict[i]['topic'] = llm_output
            logging.info(f'Topic generated for cluster {i}: {llm_output}')
        self.df_gender_filtered.to_csv('Cluster_Embedding.csv')

        return cluster_dict

if __name__ == "__main__":
    openai_api_key = ''
    
    cluster_analyzer = TwitterClusterAnalyzer(openai_api_key)
    
    cluster_analyzer.load_data('../data/cleaned_data.csv')
    
    cluster_analyzer.encode_text_embeddings()
    
    cluster_analyzer.cluster_tweets()
    
    cluster_dict = cluster_analyzer.generate_topic_titles()
    
    cluster_dict = cluster_analyzer.generate_topic_titles_with_prompt(cluster_dict)
    
    for cluster, data in cluster_dict.items():
        print(f"Cluster {cluster} - Topic Title: {data['topic']}")
        print("Cluster Documents:")
        for i, doc in enumerate(data['doc']):
            print(f"{i + 1}. {doc}")
        print("\n")

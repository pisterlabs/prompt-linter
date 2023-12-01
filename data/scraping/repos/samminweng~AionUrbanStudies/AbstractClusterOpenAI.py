# Cluster the document using OpenAI model
# Ref: https://openai.com/blog/introducing-text-and-code-embeddings/
import os
import sys
from argparse import Namespace
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import hdbscan
import umap
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import openai, numpy as np

from sklearn.metrics import pairwise_distances, silhouette_samples

openai.organization = "org-yZnUvR0z247w0HQoS6bMJ0WI"
openai.api_key = os.getenv("OPENAI_API_KEY")


# print(openai.Model.list())


class AbstractClusterOpenAI:
    def __init__(self, _iteration, _cluster_no):
        self.args = Namespace(
            case_name='AIMLUrbanStudyCorpus',
            embedding_name='OpenAIEmbedding',
            model_name='curie',
            iteration=_iteration,
            cluster_no=_cluster_no,
            iteration_folder='iteration_' + str(_iteration),
            cluster_folder='cluster_' + str(_cluster_no),
            phase='abstract_clustering_phase',
            path='data',
            threshold=50,   # Maximal number of abstracts in a cluster
            seed=3,
            n_neighbors=150,
            min_dist=0.0,
            epilson=0.0,
            dimensions=[500, 450, 400, 350, 300, 250, 200, 150, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55,
                        50, 45, 40, 35, 30, 25, 20],
            min_cluster_size=[10, 15, 20, 25, 30, 35, 40, 45, 50]
        )
        path = os.path.join('data', self.args.case_name + '_' + self.args.embedding_name, self.args.iteration_folder,
                            self.args.cluster_folder, self.args.case_name + '_cleaned.csv')
        self.text_df = pd.read_csv(path)
        # # # # # Load all document vectors without outliers
        self.text_df['Text'] = self.text_df['Title'] + ". " + self.text_df['Abstract']
        # Filter out dimensions > the length of text df
        self.args.dimensions = list(filter(lambda d: d < len(self.text_df) - 5, self.args.dimensions))

    # Get doc vectors from OpenAI embedding API
    def get_doc_vectors(self, is_load=True):
        def clean_sentence(_sentences):
            # Preprocess the sentence
            cleaned_sentences = list()  # Skip copy right sentence
            for sentence in _sentences:
                if u"\u00A9" not in sentence.lower() and 'licensee' not in sentence.lower() \
                        and 'copyright' not in sentence.lower() and 'rights reserved' not in sentence.lower():
                    try:
                        cleaned_words = word_tokenize(sentence.lower())
                        # Keep alphabetic characters only and remove the punctuation
                        cleaned_sentences.append(" ".join(cleaned_words))  # merge tokenized words into sentence
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
            return cleaned_sentences

        try:
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                                  'abstract_clustering_phase', 'doc_vectors')
            if is_load:
                path = os.path.join(folder, 'doc_vectors.json')
                # Load doc vectors
                doc_vector_df = pd.read_json(path)
                cluster_doc_ids = self.text_df['DocId'].tolist()
                cluster_doc_vector = doc_vector_df[doc_vector_df['DocId'].isin(cluster_doc_ids)]
                # print(cluster_doc_vector)
                self.text_df['DocVectors'] = cluster_doc_vector['DocVectors'].tolist()
                # # Print out the doc vector
                # print(self.text_df)
                folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                                      'abstract_clustering_phase', self.args.iteration_folder,
                                      self.args.cluster_folder, 'doc_vectors')
                Path(folder).mkdir(parents=True, exist_ok=True)
                path = os.path.join(folder, 'doc_vectors.json')
                self.text_df.to_json(path, orient='records')
            else:
                # Collect all the texts
                cleaned_texts = list()
                # Search all the subject words
                for i, row in self.text_df.iterrows():
                    try:
                        sentences = clean_sentence(sent_tokenize(row['Text']))  # Clean the sentences
                        cleaned_text = " ".join(sentences)
                        cleaned_texts.append(cleaned_text)
                    except Exception as _err:
                        print("Error occurred! {err}".format(err=_err))
                self.text_df['CleanText'] = cleaned_texts
                resp = openai.Embedding.create(
                    input=cleaned_texts,
                    engine="text-similarity-" + self.args.model_name + "-001")
                doc_embeddings = list()
                for doc_embedding in resp['data']:
                    doc_embeddings.append(doc_embedding['embedding'])
                print(doc_embeddings)
                self.text_df['DocVectors'] = doc_embeddings
                # Print out the doc vector
                print(self.text_df)
                Path(folder).mkdir(parents=True, exist_ok=True)
                path = os.path.join(folder, 'doc_vectors.json')
                self.text_df.to_json(path, orient='records')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Experiment UMAP + HDBSCAN clustering and evaluate the clustering results with 'Silhouette score'
    def run_HDBSCAN_cluster_experiments(self):
        # Calculate Silhouette score
        # Ref: https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
        # Ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
        def compute_Silhouette_score(_cluster_labels, _cluster_vectors, _cluster_results):
            # score = 1 indicates good clusters that each cluster distinguishes from other clusters
            # score = 0 no difference between clusters
            # score = -1 clusters are wrong
            try:
                # start = datetime.now()
                # Get silhouette score for each cluster
                silhouette_scores = silhouette_samples(_cluster_vectors, _cluster_labels, metric='cosine')
                avg_scores = list()
                # Get each individual cluster's score
                for _cluster_result in _cluster_results:
                    cluster = _cluster_result['cluster']
                    cluster_silhouette_scores = silhouette_scores[np.array(cluster_labels) == cluster]
                    cluster_score = np.mean(cluster_silhouette_scores)
                    _cluster_result['score'] = cluster_score
                    avg_scores.append(cluster_score)
                    # end = datetime.now()
                avg_scores = np.mean(avg_scores)
                # difference = (end - start).total_seconds()
                # print("Time difference {d} second".format(d=difference))
                return _cluster_results, avg_scores
            except Exception as err:
                print("Error occurred! {err}".format(err=err))
                return -1

        # Collect clustering results and find outliers and the cluster of minimal size
        def collect_cluster_results(_doc_vectors, _cluster_labels):
            try:
                _results = list()
                for _doc, _label in zip(_doc_vectors, _cluster_labels):
                    _doc_id = _doc['DocId']
                    _found = next((r for r in _results if r['cluster'] == _label), None)
                    if not _found:
                        _results.append({'cluster': _label, 'doc_ids': [_doc_id]})
                    else:
                        _found['doc_ids'].append(_doc_id)
                _results = sorted(_results, key=lambda c: c['cluster'], reverse=True)
                # Add the count
                for _result in _results:
                    _result['count'] = len(_result['doc_ids'])
                    _result['doc_ids'] = _result['doc_ids']
                return _results
            except Exception as c_err:
                print("Error occurred! {err}".format(err=c_err))

        # Load doc vectors
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name,
                              self.args.phase, self.args.iteration_folder, self.args.cluster_folder, 'doc_vectors')
        path = os.path.join(folder, 'doc_vectors.json')
        doc_vector_df = pd.read_json(path)
        doc_vectors = doc_vector_df.to_dict("records")
        # Doc vectors from OpenAI is 4,096
        print("OpenAI dimension {d}".format(d=len(doc_vector_df['DocVectors'].tolist()[0])))
        # Experiment HDBSCAN clustering with different parameters
        results = list()
        max_score = 0.0
        for dimension in self.args.dimensions:
            if dimension <= 500:
                # Run HDBSCAN on reduced dimensional vectors
                reduced_vectors = umap.UMAP(
                    n_neighbors=self.args.n_neighbors,
                    min_dist=self.args.min_dist,
                    n_components=dimension,
                    random_state=self.args.seed,
                    metric="cosine").fit_transform(doc_vector_df['DocVectors'].tolist())
            else:
                # Run HDBSCAN on raw vectors
                reduced_vectors = np.vstack(doc_vector_df['DocVectors'])  # Convert to 2D numpy array
            # print(reduced_vectors)
            # for min_samples in self.args.min_samples:
            epsilon = self.args.epilson
            min_samples = 1
            for min_cluster_size in self.args.min_cluster_size:
                result = {'dimension': dimension,
                          'min_cluster_size': min_cluster_size,
                          'avg_score': None, 'total_clusters': None,
                          }
                try:
                    # Compute the cosine distance/similarity for each doc vectors
                    distances = pairwise_distances(reduced_vectors, metric='cosine')
                    # Cluster reduced vectors using HDBSCAN
                    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                                     min_samples=min_samples,
                                                     cluster_selection_epsilon=epsilon,
                                                     metric='precomputed').fit_predict(
                        distances.astype('float64')).tolist()
                    # Aggregate the cluster results
                    cluster_results = collect_cluster_results(doc_vectors, cluster_labels)
                    # Sort cluster result by count

                    # Compute silhouette score for clustered results
                    distance_vectors = distances.tolist()
                    # Store the results at least 1 clusters
                    if len(cluster_results) > 1:
                        cluster_results, avg_score = compute_Silhouette_score(cluster_labels, distance_vectors,
                                                                              cluster_results)
                        outlier = next(r for r in cluster_results if r['cluster'] == -1)
                        result['avg_score'] = avg_score
                        result['total_clusters'] = len(cluster_results)
                        result['outlier'] = outlier['count']
                        result['cluster_results'] = cluster_results
                        if max_score <= avg_score:
                            result['reduced_vectors'] = reduced_vectors.tolist()
                            max_score = avg_score
                        results.append(result)
                except Exception as _err:
                    print("Error occurred! {err}".format(err=_err))
                    sys.exit(-1)
            # print(result)

        # Output the clustering results of a dimension
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase,
                              self.args.iteration_folder, self.args.cluster_folder, 'hdbscan_experiments')
        Path(folder).mkdir(parents=True, exist_ok=True)
        # Output the detailed clustering results
        result_df = pd.DataFrame(results)
        # Output cluster results to CSV
        path = os.path.join(folder, 'cluster_doc_vector_results.csv')
        result_df.to_csv(path, encoding='utf-8', index=False, columns=['dimension', 'min_cluster_size',
                                                                       'avg_score', 'total_clusters', 'outlier',
                                                                       'cluster_results'])
        path = os.path.join(folder, 'cluster_doc_vector_results.json')
        result_df.to_json(path, orient='records')

    # Get the HDBSCAN clustering results with the highest silhouette scores and plots the clustering dot chart
    def find_best_HDBSCAN_cluster_result(self):
        def visualise_cluster_results(_docs, _cluster_results, _dimension, _min_cluster_size, _folder):
            try:
                df = pd.DataFrame(_docs)
                # Visualise HDBSCAN clustering results using dot chart
                colors = sns.color_palette('tab10', n_colors=10).as_hex()
                marker_size = 8
                # Plot clustered dots and outliers
                fig = go.Figure()
                none_outliers = list(filter(lambda r: r['count'] <= 40, _cluster_results))
                for _result in none_outliers:
                    _cluster_no = _result['cluster']
                    dots = df.loc[df['Cluster'] == _cluster_no, :]
                    marker_color = colors[_cluster_no]
                    marker_symbol = 'circle'
                    name = 'Cluster {no}'.format(no=_cluster_no)
                    fig.add_trace(go.Scatter(
                        name=name,
                        mode='markers',
                        x=dots['x'].tolist(),
                        y=dots['y'].tolist(),
                        marker=dict(line_width=1, symbol=marker_symbol,
                                    size=marker_size, color=marker_color)
                    ))
                # Figure layout
                fig.update_layout(width=600, height=800,
                                  legend=dict(orientation="v"),
                                  margin=dict(l=20, r=20, t=30, b=40))
                file_name = 'dimension_' + str(_dimension) + '_min_cluster_size_' + str(_min_cluster_size)
                file_path = os.path.join(folder, file_name + ".png")
                pio.write_image(fig, file_path, format='png')
                print("Output the images of clustered results to " + file_path)
            except Exception as err:
                print("Error occurred! {err}".format(err=err))

        try:
            # Find the best results in each dimension
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase,
                                  self.args.iteration_folder, self.args.cluster_folder, 'hdbscan_experiments')
            # Load experiment results
            path = os.path.join(folder, 'cluster_doc_vector_results.json')
            results = pd.read_json(path).to_dict("records")
            # sort results by scores and min_cluster_size
            results = sorted(results, key=lambda r: (r['avg_score'], r['min_cluster_size']), reverse=True)
            # print(results)
            best_result = results[0]
            # # Get the highest score of d_results
            dimension = best_result['dimension']
            min_cluster_size = best_result['min_cluster_size']
            cluster_results = best_result['cluster_results']
            # Sort cluster results by score
            cluster_results = sorted(cluster_results, key=lambda r: r['score'], reverse=True)
            reduced_vectors = best_result['reduced_vectors']
            # Assign cluster results to docs
            docs = self.text_df.to_dict("records")
            cluster_id = 1
            for result in cluster_results:
                result['cluster'] = cluster_id
                doc_ids = result['doc_ids']
                cluster_docs = filter(lambda d: d['DocId'] in doc_ids, docs)
                for doc in cluster_docs:
                    doc['Cluster'] = cluster_id
                cluster_id = cluster_id + 1
            # print(docs)
            # Updated doc's x and y from reduced vectors
            for doc, doc_vectors in zip(docs, reduced_vectors):
                # Project the doc vectors x, y dimension for visualisation
                doc['x'] = doc_vectors[0]
                doc['y'] = doc_vectors[1]
            visualise_cluster_results(docs, cluster_results, dimension, min_cluster_size, folder)
            # Output cluster results
            df = pd.DataFrame(cluster_results)
            path = os.path.join(folder, 'cluster_results.csv')
            df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'cluster_results.json')
            df.to_json(path, orient='records')
            # Output abstract clustering
            docs_df = pd.DataFrame(docs, columns=['Cluster', 'DocId', 'Cited by', 'Title', 'Author Keywords',
                                                  'Abstract', 'Year', 'Source title', 'Authors', 'DOI',
                                                  'Document Type', 'x', 'y'])
            path = os.path.join(folder, 'docs_cluster_results.csv')
            docs_df.to_csv(path, encoding='utf-8', index=False)
            path = os.path.join(folder, 'docs_cluster_results.json')
            docs_df.to_json(path, orient='records')
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # #Output large clusters (>threshold) and store as a corpus as input for the next iteration
    def output_large_clusters_as_corpus(self):
        try:
            # Get the outliers identified by HDBSCAN
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase,
                                  self.args.iteration_folder, self.args.cluster_folder, 'hdbscan_experiments')
            path = os.path.join(folder, 'cluster_results.json')
            # Get the best clustering of silhouette score
            cluster_results = pd.read_json(path).to_dict("records")
            # Get all large clusters
            large_clusters = list(filter(lambda c: c['count'] >= self.args.threshold, cluster_results))
            next_iteration = self.args.iteration + 1
            # Load the docs
            path = os.path.join(folder, 'docs_cluster_results.json')
            docs = pd.read_json(path).to_dict("records")
            # print(large_clusters)
            for cluster in large_clusters:
                cluster_id = cluster['cluster']
                doc_ids = cluster['doc_ids']
                cluster_docs = list(filter(lambda d: d['DocId'] in doc_ids, docs))
                # print(cluster_docs)
                cluster_docs_df = pd.DataFrame(cluster_docs)
                # output to data folder
                folder = os.path.join('data', self.args.case_name + '_' + self.args.embedding_name,
                                      # self.args.cluster_folder,
                                      'iteration_' + str(next_iteration), 'cluster_' + str(cluster_id))
                Path(folder).mkdir(parents=True, exist_ok=True)
                path = os.path.join(folder, self.args.case_name + '_cleaned.csv')
                # Save outlier df to another corpus
                cluster_docs_df.to_csv(path, encoding='utf-8', index=False)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))

    # Collect all iterative abstract cluster results
    def collect_iterative_cluster_results(self):
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase)
        results = list()
        max_iteration = 4
        cluster_id = 1
        corpus = list()
        # Go through each iteration 1 to last iteration
        for i in range(1, max_iteration + 1):
            try:
                iteration_folder = os.path.join(folder, 'iteration_' + str(i))
                # Get child folder ordered by name
                cluster_folders = sorted(os.listdir(iteration_folder))
                for folder_name in cluster_folders:
                    cluster_folder = os.path.join(iteration_folder, folder_name)
                    # print(cluster_folder)
                    # Get the cluster results
                    path = os.path.join(cluster_folder, 'hdbscan_experiments', 'cluster_results.json')
                    # Load the cluster results
                    cluster_results = pd.read_json(path).to_dict("records")
                    # Filter out large clusters > 40
                    cluster_results = list(filter(lambda r: r['count'] < self.args.threshold, cluster_results))
                    # Load clustered docs result
                    path = os.path.join(cluster_folder, 'hdbscan_experiments', 'docs_cluster_results.json')
                    docs = pd.read_json(path).to_dict("records")
                    # Get summary of cluster topics
                    # print(cluster_results)
                    for cluster_result in cluster_results:
                        doc_ids = cluster_result['doc_ids']
                        results.append({
                            "iteration": i, "cluster": cluster_id, "score": cluster_result['score'],
                            "count": cluster_result['count'], "doc_ids": cluster_result['doc_ids']
                        })
                        # Get the clustered docs
                        cluster_docs = list(filter(lambda d: d['DocId'] in doc_ids, docs))
                        # Include to corpus
                        corpus.extend(cluster_docs)
                        cluster_id = cluster_id + 1
            except Exception as _err:
                print("Error occurred! {err}".format(err=_err))
                sys.exit(-1)
        print(results)
        # # Assign group no to clusters
        groups = [range(1, 6), range(6, 9), range(9, 12), range(12, 25)]
        for i, group in enumerate(groups):
            group_clusters = list(filter(lambda r: r['cluster'] in group, results))
            for cluster in group_clusters:
                cluster['group'] = i
        # # Load the results as data frame
        df = pd.DataFrame(results)
        # Output cluster results to CSV
        folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase)
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(folder, self.args.case_name + '_iterative_clustering_summary.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, self.args.case_name + '_iterative_clustering_summary.json')
        df.to_json(path, orient='records')
        # # Assign clusters to docs
        for result in results:
            cluster_id = result['cluster']
            doc_ids = result['doc_ids']
            docs = list(filter(lambda d: d['DocId'] in doc_ids, corpus))
            for doc in docs:
                doc['Cluster'] = cluster_id
        corpus = sorted(corpus, key=lambda d: d['Cluster'])
        # Output doc clusters to corpus
        df = pd.DataFrame(corpus)
        path = os.path.join(folder, self.args.case_name + '_clusters.csv')
        df.to_csv(path, encoding='utf-8', index=False)
        path = os.path.join(folder, self.args.case_name + '_clusters.json')
        df.to_json(path, orient='records')
        # print(df)

    # Plot the abstract cluster results
    def visualise_abstract_cluster_results(self):
        try:
            folder = os.path.join('output', self.args.case_name + '_' + self.args.embedding_name, self.args.phase)
            # Load clustered docs
            path = os.path.join(folder, self.args.case_name + '_clusters.json')
            corpus_df = pd.read_json(path)
            # Load cluster results
            path = os.path.join(folder, self.args.case_name + '_iterative_clustering_summary.json')
            cluster_results = pd.read_json(path).to_dict("records")
            # Visualise HDBSCAN clustering results using dot chart
            colors = sns.color_palette('Set2', n_colors=4).as_hex()
            marker_size = 8
            # Plot clustered dots and outliers
            fig = go.Figure()
            for result in cluster_results:
                cluster_id = result['cluster']
                dots = corpus_df.loc[corpus_df['Cluster'] == cluster_id, :]
                group_no = result['group']
                marker_color = colors[group_no]
                marker_symbol = 'circle'
                name = 'Cluster {no}'.format(no=cluster_id)
                fig.add_trace(go.Scatter(
                    name=name,
                    mode='markers',
                    x=dots['x'].tolist(),
                    y=dots['y'].tolist(),
                    marker=dict(line_width=1, symbol=marker_symbol,
                                size=marker_size, color=marker_color)
                ))
            # Figure layout
            fig.update_layout(width=600, height=800,
                              legend=dict(orientation="v"),
                              margin=dict(l=20, r=20, t=30, b=40))
            file_name = "abstract_cluster_dot_chart"
            file_path = os.path.join(folder, file_name + ".png")
            pio.write_image(fig, file_path, format='png')
            print("Output the images of clustered results to " + file_path)
        except Exception as err:
            print("Error occurred! {err}".format(err=err))


# Main entry
if __name__ == '__main__':
    try:
        # Re-cluster large cluster into sub-clusters
        iteration = 4
        cluster_no = 6
        ac = AbstractClusterOpenAI(iteration, cluster_no)
        # ac.get_doc_vectors(is_load=True)
        # ac.run_HDBSCAN_cluster_experiments()
        # ac.find_best_HDBSCAN_cluster_result()
        # ac.output_large_clusters_as_corpus()

        # Aggregate iterative clustering results
        ac.collect_iterative_cluster_results()
        ac.visualise_abstract_cluster_results()
    except Exception as err:
        print("Error occurred! {err}".format(err=err))

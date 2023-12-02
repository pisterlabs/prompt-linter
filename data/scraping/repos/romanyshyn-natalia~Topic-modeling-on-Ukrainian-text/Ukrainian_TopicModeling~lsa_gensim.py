from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from gensim.matutils import corpus2csc, corpus2dense
from sklearn.cluster import KMeans
import umap.umap_ as umap
import numpy as np
from pprint import pprint
import matplotlib.colors as m_colors
from collections import Counter
import pandas as pd

from preprocessor import preprocess_normalized
from vectorizer import vectorize_corpus_tf_idf

SOME_FIXED_SEED = 42


def create_gensim_lsa_model(document_term_matrix, corpus_dictionary, number_of_topics, words):
    """
    Create LSA model using gensim
    """
    np.random.seed(SOME_FIXED_SEED)
    lsa_model = LsiModel(document_term_matrix, num_topics=number_of_topics, id2word=corpus_dictionary)  # train model
    pprint(lsa_model.print_topics(num_topics=number_of_topics, num_words=words))
    return lsa_model


def compute_coherence_values(corpus_dict, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    """
    coherence_values = []
    model_list = []

    np.random.seed(SOME_FIXED_SEED)
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=corpus_dict)  # train model
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=doc_clean, dictionary=corpus_dict, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
        print(f"For {num_topics} topics: {coherence_model.get_coherence()} coherence score.")
    return model_list, coherence_values


def plot_graph(doc_clean, start, stop, step):
    dct, doc_term_matrix = vectorize_corpus_tf_idf(clean_text)
    model_list, coherence_values = compute_coherence_values(dct, doc_term_matrix, doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.savefig('./resulting_plots/lsa/coherence_measure_graph_lsa_normalized.png')


def get_vectorized_sparse_matrix(gensim_vectorized, dct, num_of_topics):
    lsa_model = LsiModel(gensim_vectorized, num_topics=num_of_topics, id2word=dct)
    # compute (m ⨉ t) document-topic matrix
    V = corpus2dense(lsa_model[gensim_vectorized], len(lsa_model.projection.s)).T / lsa_model.projection.s
    # create topics matrix for clustering plot
    X_topics = V * lsa_model.projection.s
    # convert vertorized gensim matrix to sparse matrix
    return X_topics, corpus2csc(gensim_vectorized).transpose()


def create_k_means_model(num_of_clusters, vectorized_sparse_matrix):
    """
    Create k-means clustering model using sklearn
    """
    km = KMeans(n_clusters=num_of_clusters, random_state=SOME_FIXED_SEED)
    km.fit(vectorized_sparse_matrix)
    return km.labels_.tolist()


def plot_clusters_with_topics(topics_matrix, clusters):
    """
    Visualise topics using clusters
    """
    embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=100).fit_transform(topics_matrix)
    plt.figure(figsize=(7, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=clusters,
                s=10,  # size
                edgecolor='none')
    plt.savefig('./resulting_plots/lsa/topics_clustering_lsa_normalized.png')


def align_yaxis(ax1, v1, ax2, v2):
    """
    adjust ax2 y limit so that v2 in ax2 is aligned to v1 in ax1
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)


if __name__ == "__main__":
    clean_text = preprocess_normalized()

    # create baseline model for LSA topic modeling
    dictionary, term_doc_matrix = vectorize_corpus_tf_idf(clean_text)
    base_model = create_gensim_lsa_model(term_doc_matrix, dictionary, 4, 10)

    # hyperparameter tuning
    # compute coherence score for different number of topics and plot graph with results
    # plot_graph(clean_text, 2, 21, 1)

    # optimal_number_of_topics = 5
    #
    # # create final model with optimized number of topics and visualize with the help of clusters and UMAP
    # x_topics, tf_idf_sparse = get_vectorized_sparse_matrix(term_doc_matrix, dictionary, optimal_number_of_topics)
    # cluster_labels = create_k_means_model(optimal_number_of_topics, tf_idf_sparse)
    # plot_clusters_with_topics(x_topics, cluster_labels)

    topics = base_model.show_topics(formatted=False)
    data_flat = [w for w_list in clean_text for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=160)
    cols = [color for name, color in m_colors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(-0.5, 0.5)
        ax.set_ylim(0, 800)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        align_yaxis(ax_twin, 0, ax, 0)

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords LSA', fontsize=22, y=1.05)
    plt.savefig('./resulting_plots/lsa/importance_of_topic_keywords_lsa.png')

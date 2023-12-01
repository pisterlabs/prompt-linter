import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import VectorStore
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import plotly.subplots as sp
from sklearn.decomposition import PCA
import re


def visualize_vector_search(vectorstore: VectorStore, query: str, embeddings=None):
    """
    Visualize the vector search with PCA and t-SNE.
    :param vectorstore: LangChain VectorStore
    :param query: unembedded str query
    :param embeddings: Embedding system (defaults to OpenAI)
    :return: None (opens browser window with visualization using plotly)
    """
    # visualize search results
    # embed task vector
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    embedded_query = embeddings.embed_query(query)

    # document contents (escape chars that break plotly)
    doc_contents = [d.page_content[:100] for d in vectorstore.docstore._dict.values()] + [query]
    chars_to_remove = "<>/'\"`"  # Not sure which chars exactly break it
    pattern = "[" + re.escape(chars_to_remove) + "]"
    doc_contents = [re.sub(pattern, '', s) for s in doc_contents]

    visualize_faiss_index_with_query(vectorstore.index, embedded_query, doc_contents, k=4)


def visualize_faiss_index_with_query(index, query_vector, doc_texts, k=4):
    """
    Visualize the vector search with PCA and t-SNE.

    :param index: FAISS index
    :param query_vector: embedded vector to search for
    :param doc_texts: list of document texts for hover text (must be in same order as index)
    :param k: number of closest vectors to show
    :return: None (opens browser window with visualization using plotly)
    """
    # Search for the k closest vectors to the query vector
    distances, indices = index.search(np.array([query_vector]), k)

    # Retrieve all the vectors from the FAISS index
    retrieved_vectors = index.reconstruct_n(0, index.ntotal)

    # Apply PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    all_data_pca = pca.fit_transform(np.vstack([retrieved_vectors, query_vector]))

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    all_data_tsne = tsne.fit_transform(np.vstack([retrieved_vectors, query_vector]))

    # Create a DataFrame for the PCA and t-SNE data and assign labels and add text descriptions
    data_pca_df = pd.DataFrame(all_data_pca, columns=['PCA 1', 'PCA 2'])
    data_tsne_df = pd.DataFrame(all_data_tsne, columns=['t-SNE 1', 't-SNE 2'])
    data_pca_df['label'] = data_tsne_df['label'] = 'Other'
    data_pca_df.loc[indices[0], 'label'] = data_tsne_df.loc[indices[0], 'label'] = 'Close'
    data_pca_df.loc[data_pca_df.index[-1], 'label'] = data_tsne_df.loc[data_tsne_df.index[-1], 'label'] = 'Query'
    data_pca_df['doc_text'] = data_tsne_df['doc_text'] = doc_texts

    # Create subplots
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('PCA', 't-SNE'))

    # Create PCA scatter plot and add to subplot
    pca_scatter = px.scatter(data_frame=data_pca_df, x='PCA 1', y='PCA 2', color='label',
                             color_discrete_sequence=['blue', 'magenta', 'red'],
                             hover_data={'doc_text': True, 'label': False, 'PCA 1': False, 'PCA 2': False},
                             )

    for trace in pca_scatter.data:
        fig.add_trace(trace, row=1, col=1)

    # Create t-SNE scatter plot and add to subplot
    tsne_scatter = px.scatter(data_frame=data_tsne_df, x='t-SNE 1', y='t-SNE 2', color='label',
                              color_discrete_sequence=['blue', 'magenta', 'red'],
                              hover_data={'doc_text': True, 'label': False, 't-SNE 1': False, 't-SNE 2': False},
                              )

    for trace in tsne_scatter.data:
        fig.add_trace(trace, row=1, col=2)

    pca_scatter.update_traces(
        marker=dict(size=[5 for i in range(len(data_pca_df))],
                    opacity=[0.5 for i in range(len(data_pca_df))]),
        selector=dict(type='scattergl'))

    tsne_scatter.update_traces(
        marker=dict(size=[5 for i in range(len(data_tsne_df))],
                    opacity=[0.5 for i in range(len(data_tsne_df))]),
        selector=dict(type='scattergl'))

    # Update the layout and show the plot
    fig.update_layout(title='Dimensionality Reduction Visualization of Vectors from a FAISS Index with Query Vector')
    fig.show()

if __name__ == '__main__':
    import faiss

    # Generate 100-dimensional random vectors
    n_vectors = 500
    dim = 1882
    data = np.random.rand(n_vectors, dim).astype('float32')

    # Create a FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(data)

    # Generate a random query vector
    query_vector = np.random.rand(dim).astype('float32')

    # Create hover-over labels for the vectors
    doc_texts = [f'Document #{i} Contents' for i in range(n_vectors)] + ['Query Contents']

    # Visualize the FAISS index with the query vector
    visualize_faiss_index_with_query(index, query_vector, doc_texts)
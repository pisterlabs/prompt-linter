import pandas as pd
from llama_index import StorageContext, load_index_from_storage
from llamaindex_utils import set_service_context
import matplotlib.pyplot as plt
import numpy as np

def get_all_scores(query_results, show_hist=True):
    '''
    query_results is the list of nodes returned by retriever.retrieve()
    '''
    scores = [node.score for node in query_results]
    if show_hist:
        vals = np.percentile(scores, [50, 99, 99.7])
        plt.hist(scores, bins=50)
        for val in vals:
            plt.axvline(val, color='r', linestyle='--')
        plt.show()
    return scores

def gen_retriever(index_name, data_name, openai_api_key=None, path_to_db_folder='../data/'):
    '''
    index_name: 'VectorStoreIndex', 'SimpleKeywordTableIndex', 'RAKEKeywordTableIndex'
    data_name: 'citation', 'abstract'
    Input your own openai_api_key or it's going to detect environment variable OPENAI_API_KEY.
    '''
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=path_to_db_folder+f'/llamaindex{index_name}_openaiEmbed_{data_name}_db/'),
        service_context=set_service_context(openai_api_key)
    )
    retriever = index.as_retriever()
    n = len(index.docstore.docs) # just grab all similarity scores
    if index_name == 'VectorStoreIndex':
        retriever.similarity_top_k = n
    else:
        retriever.num_chunks_per_query = n # there won't be a score in the case of keyword table index though
    return retriever

def rearrange_query_results(query_results, score_threshold=0.9, sort=True):
    '''
    If using VectorStoreIndex, keep the query results with similarity score > score_threshold.
    Then count the number of times each doc appears in the query results and combine the reasons of citation.
    Sort the query results by the number of times each doc appears.
    Return the sorted docs.
    If using keyword table index, just rearrange the query results into a pd.DataFrame.
    '''
    if query_results[0].score is None: # keyword table index
        rst = pd.DataFrame(columns=['doc_id', 'reasons'])
        for node in query_results:
            doc_id = int(node.metadata['doc_id'])
            rst.loc[len(rst)] = {'doc_id': doc_id, 'reasons': node.text}
        return rst

    # collect results
    rst = pd.DataFrame(columns=['doc_id', 'n', 'reasons'])
    for node in query_results:
        if node.score < score_threshold:
            break
        doc_id = int(node.metadata['doc_id'])
        if doc_id in rst['doc_id'].values:
            rst.loc[rst['doc_id']==doc_id, 'n'] += 1
            rst.loc[rst['doc_id']==doc_id, 'reasons'] += '; ' + node.text
        else:
            rst.loc[len(rst)] = {'doc_id': doc_id, 'n': 1, 'reasons': node.text}

    # sort results
    if sort:
        rst = rst.sort_values(by='n', ascending=False, kind='mergesort')
    return rst

def print_citation_query_results(data_citation, query_results, i_start=0, i_end=5, data_abstract=None):
    '''
    data_citation: '../data/FRB_citations.csv'
    query_results: the output of rearrange_citation_query_results()
    i_start, i_end: start and end indices of the sorted_results to print out
    if data_abstract ('../data/FRB_abstracts.csv') is not None, print out title and abstract as well
    '''
    if i_start >= len(query_results):
        return
    i_end = min(i_end, len(query_results))
    # print out full name of reference, arxiv id, and reasons for citation
    for i in range(i_start, i_end):
        doc_id = query_results.iloc[i]['doc_id']
        reasons = query_results.iloc[i]['reasons']
        if len(reasons) > 1000: # truncate if too long
            reasons = reasons[:1000] + '...'
        print(f'{i+1}. {data_citation.iloc[doc_id]["txt_ref"]}, {data_citation.iloc[doc_id]["arxiv_id"]}\n')
        print(f'Reasons of citation: {reasons}\n')
        # title and abstract
        if data_abstract is not None:
            arxiv_id = data_citation.iloc[doc_id]['arxiv_id']
            title = data_abstract[data_abstract['arxiv_id']==arxiv_id]['title'].values[0]
            abstract = data_abstract[data_abstract['arxiv_id']==arxiv_id]['abstract'].values[0]
            print(f'Title: {title}\n')
            print(f'Abstract: {abstract}\n')

def print_abstract_query_results(data_abstract, query_results, i_start=0, i_end=5):
    '''
    data_abstract: '../data/FRB_abstracts.csv'
    query_results: the output of query_abstract()
    i_start, i_end: start and end indices of the query_results to print out
    '''
    if i_start >= len(query_results):
        return
    i_end = min(i_end, len(query_results))
    # print out title, abstract, authors, and arxiv id
    for i in range(i_start, i_end):
        doc_id = int(query_results.iloc[i]['doc_id'])
        title = data_abstract.iloc[doc_id]['title']
        abstract = data_abstract.iloc[doc_id]['abstract']
        arxiv_id = data_abstract.iloc[doc_id]['arxiv_id']
        authors = data_abstract.iloc[doc_id]['authors']
        print(f'{i+1}. {authors}, {arxiv_id}\n')
        print(f'Title: {title}\n')
        print(f'Abstract: {abstract}\n')

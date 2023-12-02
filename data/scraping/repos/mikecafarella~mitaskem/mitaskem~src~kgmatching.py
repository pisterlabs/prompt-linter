import pandas as pd 
import time

import os
from mitaskem.globals import CACHE_BASE
import requests
from typing import List



def make_name_doc(tup):
    (name_str, synonym_string) = tup
    if synonym_string == '':
        syns = []
    else:
        syns = synonym_string.split(';')

    if name_str == '':
        name = []
    else:
        name = [name_str]

    keywords = name + syns    
    doc = ';'.join(keywords)
    return doc

def make_desc_doc(tup):
    (name_doc, name_desc) = tup
    if name_doc != '' and name_desc != '':
        return f'{name_doc}: {name_desc}'
    elif name_doc != '':
        return name_doc
    elif name_desc != '':
        return name_desc
    else:
        return ''

def make_doc_df(df):
    """ 
    adds a 'document' column to the dataframe, which will be used for indexing
    """

    print(f'building index for KG ')
    df = df.rename({'name:string':'name', 'synonyms:string[]':'synonyms', 'id:ID':'id', 
                    'description:string':'description', 'type:string':'type'}, axis=1)
    df = df.assign(**(df[['name', 'synonyms', 'description']].fillna('')))

    df = df.assign(name_doc = df[['name', 'synonyms']].apply(tuple, axis=1).map(make_name_doc))
    df = df.assign(desc_doc = df[['name_doc', 'description']].apply(tuple, axis=1).map(make_desc_doc))
    cleandf = df[~(df.desc_doc == '')]
    return cleandf


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# def build_kg_index(document_array):
#     # not used right now. quality dropped a lot
#     import annoy

#     vectorizer = TfidfVectorizer(max_features=4096)
#     X = vectorizer.fit_transform(document_array)

#     dX = np.array(X.todense())
#     ndX = dX/(np.linalg.norm(dX, axis=1)+ 1e-6).reshape(-1,1)
#     idx = annoy.AnnoyIndex(ndX.shape[1], 'dot')
#     for i, v in enumerate(ndX):
#         idx.add_item(i, v.reshape(-1))

#     idx.build(40)
#     return vectorizer, idx

# from langchain import TFIDFRetriever, Document
# def build_node_retriever(cleandf, limit) -> TFIDFRetriever:
#     docs = cleandf['desc_doc'].values.tolist()
#     metas = cleandf[['name', 'synonyms', 'id', 'description', 'type']].apply(dict, axis=1).values.tolist()
#     as_docs = [Document(page_content=doc_search, metadata=meta) for (doc_search, meta) in zip(docs, metas)]
#     retriever = TFIDFRetriever.from_documents(as_docs, k=limit)
#     print('done building index')
#     return retriever

KG_BASE= f'{str(CACHE_BASE)}/kgs/'

_g_node_path = {
    'epi':'https://askem-mira.s3.amazonaws.com/dkg/epi/build/2023-07-07/nodes.tsv.gz',
    'climate':'https://askem-mira.s3.amazonaws.com/dkg/climate/build/2023-10-19/nodes.tsv.gz'
}

def _get_kg(kg_domain) -> pd.DataFrame:
    assert kg_domain in _g_node_path.keys()

    base = f'{KG_BASE}/{kg_domain}'
    if not os.path.exists(base):
        os.makedirs(base, exist_ok=True)

    if not os.path.exists(os.path.expanduser(f'{base}/nodes.tsv.gz')):
        url = _g_node_path[kg_domain]
        print(f'downloading kg graph {url=}')

        response = requests.get(url)
        with open(os.path.expanduser(f'{base}/nodes.tsv.gz'), 'wb') as f:
            f.write(response.content)

        print('done downloading kg graph')

    tab = pd.read_csv(os.path.expanduser(f'{base}/nodes.tsv.gz'), sep='\t', compression='gzip')
    return tab

import joblib
import numpy as np
from mitaskem.src.txfolder import transactional_folder
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

class MyRetriever2:
    def __init__(self, documents_df, vectorizer, sparse_X):
        self.documents = documents_df
        self.vectorizer = vectorizer
        self.sparse_X = sparse_X

    @staticmethod
    def build_from_df(documents_df):
        vectorizer = TfidfVectorizer()
        sparse_X = vectorizer.fit_transform(documents_df['desc_doc'].values)
        return MyRetriever2(documents_df, vectorizer, sparse_X)
    
    def save(self, base):
        with transactional_folder(base, force=True) as temp_folder:
            joblib.dump(self.vectorizer, os.path.expanduser(f'{temp_folder}/vectorizer.joblib'))
            sp.save_npz(os.path.expanduser(f'{temp_folder}/sparse_X.npz'), self.sparse_X)
            self.documents.to_parquet(f'{temp_folder}/documents.parquet')

    @staticmethod
    def load(base):
        vectorizer = joblib.load(os.path.expanduser(f'{base}/vectorizer.joblib'))
        sparse_X = sp.load_npz(os.path.expanduser(f'{base}/sparse_X.npz'))
        documents = pd.read_parquet(f'{base}/documents.parquet')
        return MyRetriever2(documents, vectorizer, sparse_X)
    
    def batch_query(self, queries, k):
        query_vec = self.vectorizer.transform(queries)
        query_vec = np.array(query_vec.todense())
        query_vec = query_vec/(np.linalg.norm(query_vec, axis=1)+ 1e-6).reshape(-1,1)

        raw_dists = cosine_similarity(query_vec, self.sparse_X)
        res = np.argsort(-raw_dists, axis=1)[:,:k]
        dists = raw_dists[np.arange(raw_dists.shape[0])[:,None], res]

        acc = []
        for (rets, dists) in zip(res, dists):
            score = 1. - np.array(dists)
            recs = self.documents.iloc[rets][['name', 'synonyms', 'id', 'description', 'type']].assign(_score=score).to_dict('records')
            acc.append(recs)

        return acc


_g_retriever_cache : dict[str,MyRetriever2] = {
    'epi': None,
    'climate': None
}

def _get_retriever(kg_domain) -> MyRetriever2:
    ''' initializes and caches retriever from kg nodes file
        use this instead of the global variable directly
    '''
    global _g_retriever_cache
    base =  f'{KG_BASE}/{kg_domain}/'

    if _g_retriever_cache.get(kg_domain) is None:
        fs_retriever_cache = f'{base}/myretriever2'

        if os.path.exists(fs_retriever_cache):
            print('loading retriever from disk cache')
            start = time.time()
            _g_retriever_cache[kg_domain] = MyRetriever2.load(fs_retriever_cache)
            print(f'done loading retriever from disk cache, {time.time() - start=}')
        else:
            print('building retriever from scratch')
            start = time.time()
            df = _get_kg(kg_domain=kg_domain)
            docdf = make_doc_df(df)
            ret = MyRetriever2.build_from_df(docdf)
            ret.save(fs_retriever_cache)
            _g_retriever_cache[kg_domain] = ret
            print('done building retriever')

        assert os.path.exists(fs_retriever_cache)

    assert _g_retriever_cache.get(kg_domain) is not None
    return _g_retriever_cache.get(kg_domain)

def local_batch_get_mira_dkg_term(term_list : List[str], kg_domain : str) -> List[dict]:
    retriever : MyRetriever2 = _get_retriever(kg_domain=kg_domain)
    return retriever.batch_query(term_list, k=4)
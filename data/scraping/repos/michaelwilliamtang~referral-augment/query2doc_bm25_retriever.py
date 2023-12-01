import numpy as np
from sqlitedict import SqliteDict
import openai

from .bm25_retriever import BM25Retriever

class Query2DocBM25Retriever(BM25Retriever):
    '''
    use_sql: use SQLite cache to reduce GPT-3 usage instead of in-memory dict
    openai_organization: string, OpenAI organization
    openai_api_key: string, OpenAI API key
    '''
    def __init__(self, *args, use_sql=False, openai_organization=None, openai_api_key=None, **kwargs):
        assert openai_organization is not None
        assert openai_api_key is not None
        openai.organization = openai_organization
        openai.api_key = openai_api_key

        # create query2doc cache to reduce LLM usage
        cache_fname = 'query2doc_cache.sqlite'
        if use_sql:
            self.query2doc_cache = SqliteDict(cache_fname)
        else:
            self.query2doc_cache = {}

        super().__init__(*args, **kwargs)

    def get_query2doc(self, query, verbose=True):
        # check cache for response
        if query in self.query2doc_cache:
            if verbose:
                print('Using cached value for query')
            return self.query2doc_cache[query]

        # if not in cache, generate
        prefix = 'Write the abstract of the paper that is cited as [CITATION] in the below passage: '
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=prefix + query,
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1,
            stop=['\n\n\n']
            )
        if verbose:
            print('Used GPT-3 to generate query')

        # cache response
        self.query2doc_cache[query] = response['choices'][0]['text']
        return response['choices'][0]['text']

    def retrieve(self, query, num_docs=10, query_weight=5, verbose=True):
        '''
        query: string
        num_docs: int, number of top documents to retrieve
        query_weight: int >= 1, weight to give original query vs. query2doc pseudo-doc
        '''
        # append query2doc augmentation
        query2doc = self.get_query2doc(query, verbose=verbose)
        query = ' '.join([query] * query_weight) + ' ' + query2doc

        return super().retrieve(query, num_docs=num_docs, query_weight=query_weight, verbose=verbose)

    # if using SQLite cache, call this to close at end of session
    def close_cache(self):
        self.query2doc_cache.commit()
        self.query2doc_cache.close()

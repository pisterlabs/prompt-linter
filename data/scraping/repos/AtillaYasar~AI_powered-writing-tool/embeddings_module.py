import os, time, json, openai, math, time
import numpy as np
from secret_things import openai_key

from overall_imports import text_append, text_create, text_read, open_json, col, make_json

openai.organization = "org-ExxER7UutRm3CU6M9FdszAoE"
openai.api_key = openai_key

def use_api(string):
    """Uses OpenAI API to retrieve ada-002 text embeddings of a string."""

    print(col('cy','using api for ') + string)
    if type(string) is not str:
        exit('use_api can only take a string')
    response = openai.Embedding.create(input=string, model='text-embedding-ada-002')
    embedding = response['data'][0]['embedding']
    return embedding

class DataHandler:
    """Indexer maps a string to the path where the embedding of that string is stored."""

    def __init__(self):
        '''
        mappings you need:
            - string --> embedding json and metadata
            - string --> index
        '''
        self.string_to_index = self.get_string_to_index()
        self.string_to_info = self.get_string_to_info()
        self.emb_array = self.get_emb_array()
        self.embedding_folder = self.get_emb_folder()

        # i want certainty about the structure.
        assert type(self.string_to_info) is dict
        for k,v in self.string_to_info.items():
            assert type(v) is dict
            for key in ['path', 'meta']:
                assert key in v
            assert type(v['path']) is str
            assert type(v['meta']) is list
        
        if not (len(self.emb_array) == len(self.string_to_index)):
            print('emb_array and string_to_index are not the same length')
            print('emb_array:', len(self.emb_array))
            print('string_to_index:', len(self.string_to_index))
            self.make_numpy_array()
        
        assert len(self.emb_array) == len(self.string_to_index) == len(self.string_to_info)
        assert self.string_to_index.keys() == self.string_to_info.keys()

        print(col('gr', 'DataHandler.init successful'))

    
    # 4 setup helpers
    def get_string_to_info(self):
        path = 'string_to_info.json'
        if path not in os.listdir():
            make_json({}, path)
        return open_json(path)
    def get_string_to_index(self):
        path = 'string_to_index.json'
        if path not in os.listdir():
            make_json({}, path)
        return open_json(path)
    def get_emb_folder(self):
        path = 'embeddings'
        if path not in os.listdir():
            os.mkdir(path)
        return path
    def get_emb_array(self):
        path = 'emb_array.npy'
        if path not in os.listdir():
            np.save(path, np.array([]))
        return np.load(path)

    def eat_data(self):
        datafolder = "collecting data for embeddings/data"
        paths = os.listdir(datafolder)
        for p in paths:
            tags = []

            ext = p.split('.')[-1]
            assert ext == 'txt'

            name = p[:-3]
            tags.append(name)
            content = text_read(f'{datafolder}/{p}')
            lines = []
            for line in content.split('\n'):
                if line.startswith('!!!'):
                    tags.append(line[3:])
                else:
                    lines.append(line)
            
            actual_content = '\n'.join(lines)
            i = input(f'path:{p}, tag:{tags}. continue?\n')
            self.get_embedding(actual_content, tags)

    def make_numpy_array(self):
        # make numpy array of embeddings
        t0 = time.time()
        embeddings_list = []
        string_to_index = {}
        for n, (string, info) in enumerate(self.string_to_info.items()):
            emb_path = info['path']
            emb = open_json(emb_path)
            embeddings_list.append(emb)
            string_to_index[string] = n
        self.emb_array = np.array(embeddings_list)
        self.string_to_index = string_to_index
        print(time.time()-t0, 'seconds to make an np array')
        np.save('emb_array.npy', self.emb_array)
        make_json(string_to_index, 'string_to_index.json')

    def _find_embedding(self, string):

        idx = self.string_to_index.get(string, None)
        if idx != None:
            emb = self.emb_array[idx]
            return 'success', emb
        else:
            indexer = self.string_to_index
            if string in indexer:
                emb = open_json(indexer[string]['path'])
                return 'success', emb
            else:
                return 'fail', None
    
    def _store_embedding(self, string, embedding, meta):
        """Will store the embedding of a string in the database."""

        assert type(string) is str
        assert type(embedding) is list
        assert type(meta) is list
        for item in meta:
            assert type(item) is str

        # make json file of embedding
        emb_path = f'{self.embedding_folder}/{time.time()}.json'
        make_json(embedding, emb_path)

        # update the mappings
        self.string_to_info[string] = {'path': emb_path, 'meta': meta}
        make_json(self.string_to_info, 'string_to_info.json')
        self.string_to_index[string] = len(self.emb_array)
        make_json(self.string_to_index, 'string_to_index.json')
        self.make_numpy_array()

        assert len(self.emb_array) == len(self.string_to_index) == len(self.string_to_info)

    def get_embedding(self, string, meta=['search term']):
        """Will return the embedding of a string.

        If the string is embedded, will just return the vector,
            if not, will use OpenAI's api to get the vector, then store it with metadata, then return the vector."""

        assert type(string) is str
        assert type(meta) is list
        for item in meta:
            assert type(item) is str

        report, emb = self._find_embedding(string)
        if report == 'success':
            return emb            
        else:
            emb = use_api(string)
            self._store_embedding(string, emb, meta)
            return emb

    def embed_list(self, lst, meta_lst):
        for item, meta in zip(lst, meta_lst):
            assert type(meta) is list
            self.get_embedding(item, meta)
    
    def delete_embedding(self, string):
        if string in self.string_to_index:
            del self.string_to_index[string]
            return True
        else:
            return False

    '''
    helpers for users:
        - get_tags() to get all tags in database
        - get_common_tags(tag) to find tags that appear together with `tag`
    '''
    def get_tags(self):
        tags = []
        for info in self.string_to_info.values():
            tags += info['meta']
        return list(set(tags))

    def get_common_tags(self, tag):
        tags = []
        for info in self.string_to_info.values():
            other_tags = [t for t in info['meta'] if t != tag]
            tags += other_tags
        return list(set(tags))

    def search(self, embedded_searchterm, search_parameters):
        """
        embedded_searchterm -- the embedding with which to compare the stuff in the database
        search_parameters -- stuff about how to shape the dataset during search
        """

        assert isinstance(embedded_searchterm, (list, np.ndarray))
        assert type(search_parameters) is dict

        for k in ['n', 'hasno', 'has']:
            assert k in search_parameters
        print(f'search params: {search_parameters}')

        top_n = search_parameters['n']
        hasno = search_parameters['hasno']
        has = search_parameters['has']

        t0 = time.time()

        assert len(self.emb_array) == len(self.string_to_index) == len(self.string_to_info)

        embeddings_iterable = self.emb_array
        scores = np.dot(self.emb_array, embedded_searchterm)  # <-- the embedding similarity scores

        strings_iterable = [string.encode('utf-8') for string in self.string_to_info.keys()]
        paths_iterable = [item['path'].encode('utf-8') for item in self.string_to_info.values()]
        metatags_iterable = [item['meta'] for item in self.string_to_info.values()]

        my_types = [
            ('score', float),
            ('embedded text', f'S{max(map(len,strings_iterable))}'),
            ('path', f'S{max(map(len,paths_iterable))}'),
            ('meta tags', list),
        ]
        # creates a neat numpy array so you can do np.sort and apply a filter mask
        combined = list(zip(scores, strings_iterable, paths_iterable, metatags_iterable))
        combined_array = np.array(combined, dtype=my_types)

        # create boolean filter mask. (i think thats what its called)
        filtermask = []
        # for creating a filter mask.
        def checker(meta_tags, has, hasno):
            # deal breakers return False. if it survives, it returns True
            for existing_tag in meta_tags:
                if existing_tag in hasno:
                    return False
            for required in has:
                if required not in meta_tags:
                    return False
            return True

        # use checker to... actually create the boolean filter mask
        print(len(metatags_iterable), len(combined_array))
        assert len(metatags_iterable) == len(combined_array)
        for item in metatags_iterable:
            if checker(item, has, hasno):
                filtermask.append(True)
            else:
                filtermask.append(False)

        assert len(filtermask) == len(combined_array)
        # apply filter mask
        combined_array = combined_array[filtermask]

        # sort by score, get top_n
        transformed = np.sort(combined_array, order='score')
        transformed = reversed(transformed[-top_n:])

        # make result usable without needing to know the specifics of the code above.
        # and the for loop is not costly anymore, with a tiny array.
        result = []
        for item in transformed:
            result.append({
                'score':round(item[0], 3),
                'text':item[1].decode('utf-8'),
                'path':item[2].decode('utf-8'),
                'meta tags':item[3],
            })

        print(f'search took {time.time()-t0} seconds')

        print(col('re', '=============================='))

        return result

    def search_and_show(self, search_term, params):
        if type(search_term) is str:
            embedding_to_search = self.get_embedding(search_term, ['search query'])
        elif type(search_term) is list:
            all_embs = []
            for item in search_term:
                assert type(item is str)
                all_embs.append(
                    self.get_embedding(item, ['search query'])
                )
            summed = all_embs[0]
            for emb in all_embs[1:]:
                for idx, scalar in enumerate(emb):
                    summed[idx] += scalar
            for idx, scalar in enumerate(summed):
                summed[idx] /= len(all_embs)
            assert len(summed) == len(all_embs[0])
            embedding_to_search = summed
        else:
            raise TypeError

        # do search
        print(f'params:{params}')
        searchres = self.search(embedding_to_search, params)

        # show results nicely.
        for item in searchres:
            s = item['score']
            te = item['text']
            ta = item['meta tags']

            print(f'score ' + col('cy', s))
            print(te)
            print(' '*4 + '--- ' + ', '.join(ta) + ' ---')
            print()



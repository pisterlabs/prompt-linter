'''
Definition and implementation of vectorizers to transform input text into a ML usable format.
'''

from abc import ABC, abstractmethod
from type_utils import ProcessedData, Label, UnprocessedData
import os
import re
from lxml import etree
from nltk.corpus import stopwords
from typing import List, Tuple, Callable
import numpy as np
import gensim
from gensim.models import KeyedVectors
import numpy.typing as npt
import fasttext
from numpy.typing import NDArray
from numpy import float32, int32
from tqdm import tqdm
import openai
import dotenv

class AbstractVectorizer(ABC):
    '''
    Abstract vectorizer class to define the interface for vectorizers.
    '''
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def vectorize(self, data: ProcessedData | UnprocessedData) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """Load and preprocess text data"""
        pass

    def _get_y(self, data: ProcessedData | UnprocessedData) -> npt.NDArray[np.int32]:
        return np.array([Label.good] * len(data['good']) + [Label.bad] * len(data['bad']))

    def _save_vectors(self, path_to_vectors: str, X: npt.NDArray[np.float32]) -> None:
        np.save(path_to_vectors, X)

    def _load_vectors(self, path_to_vectors: str) -> npt.NDArray[np.float32]:
        return np.load(path_to_vectors)
    
    def _naive_sentence_vectors(self, model, sentences: List[List[str]]):
        return np.array([np.mean([model.wv[word] for word in sentence if word in model.wv], axis=0) for sentence in sentences])
    
class PreW2V(AbstractVectorizer):
    '''
    Using pretrained W2V vectors to vectorize text. Supply a binary file to specify the model weights. Optional 'strategy' parameter to specify how to vectorize sentences from words.
    '''
    def __init__(self, path_to_bin: str, always_reload=False, strategy: int = 0) -> None:
        strat_enum = {
            0: self._naive_sentence_vectors,
            1: self._min_pooling,
            2: self._max_pooling
        }

        self.strategy: Callable[[KeyedVectors, List[List[str]]], NDArray[np.float32]] = strat_enum[strategy]
        self.always_reload = always_reload
        self.path_to_bin = path_to_bin + '.bin'
        self.vector_path = f'pt_{path_to_bin}'
        self.model = KeyedVectors.load_word2vec_format(self.path_to_bin, binary=True, unicode_errors='ignore')

    def _naive_sentence_vectors(self, model: KeyedVectors, sentences: List[List[str]]) -> npt.NDArray[np.float32]:
        return np.array([model.get_mean_vector(sentence) for sentence in sentences])
    
    def _min_pooling(self, model: KeyedVectors, sentences: List[List[str]]) -> npt.NDArray[np.float32]:
        return np.array([np.min([model[word] for word in sentence if word in model.key_to_index], axis=0) for sentence in sentences])
    
    def _max_pooling(self, model: KeyedVectors, sentences: List[List[str]]) -> npt.NDArray[np.float32]:
        return np.array([np.max([model[word] for word in sentence if word in model.key_to_index], axis=0) for sentence in sentences])

    def get_name(self) -> str:
        return self.vector_path + '.npy'

    def vectorize(self, data: ProcessedData) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        if os.path.exists(self.get_name()) and not self.always_reload:
            print('loading from memory...')
            return self._load_vectors(self.get_name()), self._get_y(data)
        
        if self.model is None:
            self.model = KeyedVectors.load_word2vec_format(self.path_to_bin, binary=True, unicode_errors='ignore')
        model: KeyedVectors = self.model

        X = self.strategy(model, data['good'] + data['bad'])

        # X = self._naive_sentence_vectors(model, data['good'] + data['bad'])
        y = self._get_y(data)

        assert X.shape[0] == y.shape[0]

        if not self.always_reload:
            VECTOR_PATH = self.get_name()
            self._save_vectors(VECTOR_PATH, X)
            print('Saved vectors to', VECTOR_PATH)

        return X, y

class FastTextVectorizer(AbstractVectorizer):
    '''
    FastText vectorizer. Supply a binary file to specify the model weights. Optional 'strategy' parameter to specify how to vectorize sentences from words.
    '''
    def __init__(self, path_to_bin: str, always_reload=False, strategy:int = 0) -> None:
        strat_enum = {
            0: self._avg_pooling,
            1: self._min_pooling,
            2: self._max_pooling
        }

        self.strategy: Callable[[fasttext.FastText._FastText, List[List[str]]], NDArray[float32]] = strat_enum[strategy]
        self.always_reload = always_reload
        self.path_to_bin = path_to_bin + '.bin'
        self.vector_path = f'ft_{path_to_bin}'
        self.model = fasttext.load_model(self.path_to_bin)

    def get_name(self) -> str:
        return self.vector_path + '.npy'

    def vectorize(self, data: ProcessedData) -> Tuple[NDArray[float32], NDArray[int32]]:
        if os.path.exists(self.get_name()) and not self.always_reload:
            print('loading from memory...')
            return self._load_vectors(self.get_name()), self._get_y(data)
        
        if self.model is None:
            print('failed to load model')

        model: fasttext.FastText._FastText = self.model

        X = self.strategy(model, data['good'] + data['bad'])

        # X = self._naive_sentence_vectors(model, data['good'] + data['bad'])
        y = self._get_y(data)

        assert X.shape[0] == y.shape[0]

        if not self.always_reload:
            VECTOR_PATH = self.get_name()
            self._save_vectors(VECTOR_PATH, X)
            print('Saved vectors to', VECTOR_PATH)

        return X, y
    
    def _avg_pooling(self, model: fasttext.FastText._FastText, sentences: List[List[str]]) -> npt.NDArray[np.float32]:
        words = model.get_words()
        return np.array([np.mean([model.get_word_vector(word) for word in sentence if word in words], axis=0) for sentence in tqdm(sentences)])
    
    def _min_pooling(self, model: fasttext.FastText._FastText, sentences: List[List[str]]) -> npt.NDArray[np.float32]:
        words = model.get_words()
        return np.array([np.min([model.get_word_vector(word) for word in sentence if word in words], axis=0) for sentence in tqdm(sentences)])
    
    def _max_pooling(self, model: fasttext.FastText._FastText, sentences: List[List[str]]) -> npt.NDArray[np.float32]:
        words = model.get_words()
        return np.array([np.max([model.get_word_vector(word) for word in sentence if word in words], axis=0) for sentence in tqdm(sentences)])
        

class W2VVectorizer(AbstractVectorizer):
    '''
    Train and apply a W2V model on our data - note: this should be done on the ENTIRE corpus if chosen, not just the training set. I would advise against using this in general and sticking with ada or a pretrained W2V
    '''
    def __init__(self) -> None:
        self.vector_path = 'w2v_vectors.npy'
        self.model = None

    def get_name(self) -> str:
        return self.vector_path
    
    def vectorize(self, data: ProcessedData) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        if os.path.exists(self.get_name()):
            print('loading from memory...')
            return self._load_vectors(self.get_name()), self._get_y(data)

        w2v_model = gensim.models.Word2Vec(data['good'] + data['bad'], min_count = 2)
        self.model = w2v_model
        # w2v_model.train(data['good'] + data['bad'], total_examples=len(data['good'] + data['bad']), epochs=10)
        # w2v_model.save('w2v_model')

        X = np.array([np.mean([w2v_model.wv[w] for w in words if w in w2v_model.wv] or [np.zeros(100)], axis=0) for words in data['good'] + data['bad']])
        y = self._get_y(data)

        assert X.shape[0] == y.shape[0]

        VECTOR_PATH = self.get_name()
        self._save_vectors(VECTOR_PATH, X)
        print('Saved vectors to', VECTOR_PATH)

        return X, y

class BOWVectorizer(AbstractVectorizer):
    '''
    Simple BOW vectorizer. Note: this should be done on the ENTIRE corpus if chosen, not just the training set. I would advise against using this in general and sticking with ada or a pretrained W2V
    '''
    def __init__(self) -> None:
        self.vector_path = 'bow_vectors.npy'

    def get_name(self) -> str:
        return self.vector_path

    def vectorize(self, data: ProcessedData) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        if os.path.exists(self.get_name()):
            print('loading from memory...')
            return self._load_vectors(self.get_name()), self._get_y(data)

        dictionary = gensim.corpora.Dictionary(data['good'] + data['bad'])
        corpus = [dictionary.doc2bow(text) for text in data['good'] + data['bad']]
        labels = [Label.good] * len(data['good']) + [Label.bad] * len(data['bad'])

        assert len(corpus) == len(labels)

        X = gensim.matutils.corpus2dense(corpus, num_terms=len(dictionary)).T
        y = self._get_y(data)

        assert X.shape[0] == y.shape[0]

        VECTOR_PATH = self.get_name()
        self._save_vectors(VECTOR_PATH, X)
        print('Saved vectors to', VECTOR_PATH)

        return X, y
    
    
class AdaVectorizer(AbstractVectorizer):
    '''Uses OAI Ada to create document level embeddings. Could alternatively use claude'''
    def __init__(self) -> None:
        self.vector_path = 'ada_vectors.npy'
        self.model = None

        api_key = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

        if api_key is None:
            raise Exception("OPENAI_API_KEY not found in .env. This may be because you have not created a .env file.")
        
        openai.api_key = api_key


    def _get_embedding(self, text: str, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding'] # type: ignore
    
    def get_name(self) -> str:
        return self.vector_path

    def _back_to_string(self, data_list: List[str]) -> str:
        string = ' '.join(data_list)
        print(string)
        return string

    def vectorize(self, data: UnprocessedData) -> Tuple[NDArray[float32], NDArray[int32]]:
        """
        params:
            data: UnprocessedData - List of preprocessed sentences FIXME: make this actually unprocessed goofy
        returns:
            X: ndarray
            y: ndarray

        TODO: This is definitely inefficient... optimize
        """
        if os.path.exists(self.get_name()):
            print('loading from memory...')
            return self._load_vectors(self.get_name()), self._get_y(data)
        
        y = self._get_y(data)

        # check if temp file exists
        if os.path.exists('ada_vecs.npy'):
            X = np.load('ada_vecs.npy')
            print('loaded from temp file')

            # find how many vectors we've already processed
            num_vecs = X.shape[0]
        else: 
            num_vecs = 0
            X = np.array([])

        # get the remaining vectors, saving to the temp file every 20 vectors
        for i in tqdm(range(num_vecs, len(data['good']) + len(data['bad']))):
            if i % 20 == 0 and i != 0 and X != np.array([]):
                np.save('ada_vecs.npy', X)
            if len(X) != 0:
                # print('booting from temp file', X.shape)
                X = np.append(X, [self._get_embedding((data['good'] + data['bad'])[i], model='text-embedding-ada-002')], axis=0)
            else:
                X = np.array([self._get_embedding((data['good'] + data['bad'])[i], model='text-embedding-ada-002')])

        VECTOR_PATH = self.get_name()
        if X == np.array([]):
            raise Exception('X is None')
        
        self._save_vectors(VECTOR_PATH, X)
        print('Saved vectors to', VECTOR_PATH)

        return X, y
import openai.encoder as oai
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models import Doc2Vec
import constants as const


class Vectorizer():        
    def vectorize(self, texts):
        pass

"""
    Uses the sentiment model trained on Amazon reviews to extract features from texts
    See: https://github.com/openai/generating-reviews-discovering-sentiment/tree/master/model
    See: https://arxiv.org/abs/1704.01444
"""
class OaiVectorizer(Vectorizer):
    def __init__(self):
        self.model = oai.Model()

    def vectorize(self, texts):
        return self.model.transform(texts)

"""
    Uses the Doc2Vec method of word embeddings to extract feature vectors from texts
    See: https://radimrehurek.com/gensim/models/doc2vec.html
    See: https://cs.stanford.edu/~quocle/paragraph_vector.pdf
"""
class Doc2VecVectorizer(Vectorizer):
    MODEL_FILE = const.FILE_DOC2VEC_MODEL

    def __init__(self):
        self.model = Doc2Vec.load(self.MODEL_FILE)
        self.stemmer = SnowballStemmer("english")
        self.lemmatizzer = WordNetLemmatizer()
    
    def vectorize(self, texts):
        processed = self.preprocess(texts)
        vectors = [self.model.infer_vector(text.split()) for text in processed]
        return vectors

    def preprocess(self, texts):
        processed = []
        for text in texts:            
            desired_tokens = []
            for token in simple_preprocess(text):
                if token not in STOPWORDS and len(token) > 3:
                    desired_tokens.append(self.lemmatize_and_stem(token))
            processed.append(' '.join(desired_tokens))
        return processed
    
    def lemmatize_and_stem(self, text):
        return self.stemmer.stem(self.lemmatizzer.lemmatize(text, pos='v'))
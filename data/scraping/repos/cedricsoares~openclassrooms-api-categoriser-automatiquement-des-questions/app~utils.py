import pandas as pd
import lxml
import html5lib
import re
import pickle
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.utils import simple_preprocess
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel

def clean_html(text):
    """
    Remove HTML from a text.
    
    Args:
        text(String): Row text with html       
    Returns:
        cleaned String
    """ 
    soup = BeautifulSoup(text, "html5lib")

    for sent in soup(['style', 'script']):
            sent.decompose()
    
    return ' '.join(soup.stripped_strings)

def text_cleaning(text):
    """
    Remove figures, punctuation, words shorter than two letters (excepted C or R) in a lowered text. 
    
    Args:
        text(String): Row text to clean  
    Returns:
        res(string): Cleaned text
    """
    pattern = re.compile(r'[^\w]|[\d_]')
    
    try: 
        res = re.sub(pattern," ", text).lower()
    except TypeError:
        return text
    
    res = res.split(" ")
    res = list(filter(lambda x: len(x)>3 , res)) #Keep singles c and r because it might be used as name of languages
    res = " ".join(res)
    return res


def tokenize(text):
    """
    Tokenize words of a text.
    
    Args:
        text(String): Row text
    Returns
        res(list): Tokenized string.
    """
    stop_words = set(stopwords.words('english'))
    
    try:
        res = word_tokenize(text, language='english')
    except TypeError:
        return text
    
    res = [token for token in res if token not in stop_words]
    return res

def filtering_nouns(tokens):
    """
    Filter singular nouns
    
    Args:
        tokens(list): A list o tokens
        
    Returns:
    
        res(list): Filtered token list
    """ 
    res = nltk.pos_tag(tokens)
    
    res = [token[0] for token in res if token[1] == 'NN']
    
    return res
    
def lemmatize(tokens):
    """
    Transform tokens into lems 
    
    Args:
        tokens(list): List of tokens       
    Returns:
        lemmatized(list): List of lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    
    for token in tokens:
        lemmatized.append(lemmatizer.lemmatize(token))
        
    return lemmatized

class SupervisedModel:

    def __init__(self):
        filename_supervised_model = "./models/svm_model.pkl"
        filename_mlb_model = "./models/mlb_model.pkl"
        filename_tfidf_model = "./models/tfidf_model.pkl"
        filename_pca_model = "./models/pca_model.pkl"
        filename_vocabulary = "./models/vocabulary.pkl"

        self.supervised_model = pickle.load(open(filename_supervised_model, 'rb'))
        self.mlb_model = pickle.load(open(filename_mlb_model, 'rb'))
        self.tfidf_model = pickle.load(open(filename_tfidf_model, 'rb'))
        self.pca_model = pickle.load(open(filename_pca_model, 'rb'))
        self.vocabulary = pickle.load(open(filename_vocabulary, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags according to a lemmatized text using a supervied model.
        
        Args:
            supervised_model(): Used mode to get prediction
            mlb_model(): Used model to detransform
        Returns:
            res(list): List of predicted tags 
        """
        input_vector = self.tfidf_model.transform(text)
        input_vector = pd.DataFrame(input_vector.toarray(), columns=self.vocabulary)
        input_vector = self.pca_model.transform(input_vector)
        res = self.supervised_model.predict(input_vector)
        res = self.mlb_model.inverse_transform(res)
        res = list({tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
        res = [tag for tag  in res if tag in text]
        
        return res
        
class LdaModel:

    def __init__(self):
        filename_model = "./models/lda_model.pkl"
        filename_dictionary = "./models/dictionary.pkl"
        self.model = pickle.load(open(filename_model, 'rb'))
        self.dictionary = pickle.load(open(filename_dictionary, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags of a preprocessed text
        
        Args:
            text(list): preprocessed text
        Returns:
            res(list): list of tags
        """
        corpus_new = self.dictionary.doc2bow(text)
        topics = self.model.get_document_topics(corpus_new)
        
        #find most relevant topic according to probability
        relevant_topic = topics[0][0]
        relevant_topic_prob = topics[0][1]
        
        for i in range(len(topics)):
            if topics[i][1] > relevant_topic_prob:
                relevant_topic = topics[i][0]
                relevant_topic_prob = topics[i][1]
                
        #retrieve associated to topic tags present in submited text
        res = self.model.get_topic_terms(topicid=relevant_topic, topn=20)
        
        res = [self.dictionary[tag[0]] for tag in res if self.dictionary[tag[0]] in text]
        
        return res


import streamlit as st
import pandas as pd
import numpy as np
import pickle



import lxml
import html5lib
import re

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
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
    
    
    soup = BeautifulSoup(text, "html5lib")
    for sent in soup(['style', 'script']):
        sent.decompose()
    return ' '.join(soup.stripped_strings)

def text_cleaning(text):
    
    pattern = re.compile(r'[^\w]|[\d_]')
    try: 
        res = re.sub(pattern," ", text).lower()
    except TypeError:
        return text
    res = res.split(" ")        
    res = list(filter(lambda x: len(x)>3 , res))
    res = " ".join(res)
    return res

def tokenize(text):
    
    stop_words = set(stopwords.words('english'))
    try:
        res = word_tokenize(text, language='english')
    except TypeError:
        return text
    res = [token for token in res if token not in stop_words]
    return res

def filtering_nouns(tokens):
    
    res = nltk.pos_tag(tokens)
    res = [token[0] for token in res if token[1] == 'NN']
    return res

def lemmatize(tokens):
    
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for token in tokens:
        lemmatized.append(lemmatizer.lemmatize(token))  
    return lemmatized

class SupervisedModel:

    def __init__(self):
        filename_supervised_model = "./model/supervised_model.pkl"
        filename_mlb_model = "./model/mlb_model.pkl"
        filename_tfidf_model = "./model/tfidf_model.pkl"
        filename_pca_model = "./model/pca_model.pkl"
        filename_vocabulary = "./model/vocabulary.pkl"

        self.supervised_model = pickle.load(open(filename_supervised_model, 'rb'))
        self.mlb_model = pickle.load(open(filename_mlb_model, 'rb'))
        self.tfidf_model = pickle.load(open(filename_tfidf_model, 'rb'))
        self.pca_model = pickle.load(open(filename_pca_model, 'rb'))
        self.vocabulary = pickle.load(open(filename_vocabulary, 'rb'))

        

    def predict_tags(self, text):
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
        filename_model = "./model/lda_model.pkl"
        filename_dictionary = "./model/dictionary.pkl"
        self.model = pickle.load(open(filename_model, 'rb'))
        self.dictionary = pickle.load(open(filename_dictionary, 'rb'))

    def predict_tags(self, text):
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




def main():
    st.title("Application de machine learning pour catégoriser automatiquement des questions")
    st.markdown("**OpenClassrooms** Projet n°5 du parcours Machine Learning réalisé en mai 2023")
    st.info("Auteur: Claude Sabardeil")
    st.markdown("_"*10)
    
    if st.checkbox("Afficher le détail de la procédure", key=False):
            st.markdown("#### Prédiction de tags effectuée en utilisant un modèle \
                        de classification supervisée et un modèle non supervisée")
            st.markdown("* **Modèle supervisé: OneVsRestClassifier(LinearSVC())**")
            st.markdown("* **Modèle non supervisé: LDA model**")
    
    #cust_input = str(st.text_input("**Saisissez votre question**"))
    cust_input = st.text_area("Write your text")
    
    
    
    if st.button("Exécuter la prédiction de tags"):
        if len(cust_input) !=0:
        
            #on prepare le texte
            text_wo_html = clean_html(cust_input)
            cleaned_text = text_cleaning(text_wo_html)
            tokenized_text = tokenize(cleaned_text)
            filtered_noun_text = filtering_nouns(tokenized_text)
            lemmatized_text = lemmatize(filtered_noun_text)
            lda_model = LdaModel()
            unsupervised_pred = list(lda_model.predict_tags(lemmatized_text))
            supervised_model = SupervisedModel()
            supervised_pred = list(supervised_model.predict_tags(lemmatized_text))
        

            tag_full = set(unsupervised_pred + supervised_pred)
            
            # afficher les résultats
            if len(tag_full) != 0:
                st.markdown("#### - Predicted tags")
                for elt in tag_full:
                    if (elt in supervised_pred) & (elt in unsupervised_pred):
                        st.markdown("<mark style='background-color: lightgray'>**" + elt + "**</mark>",
                                    unsafe_allow_html=True)
                
                    if (elt in supervised_pred) & (elt not in unsupervised_pred):
                        st.markdown("<mark style='background-color: lightblue'>**" + elt + "**</mark>",
                                    unsafe_allow_html=True)
                    
                    if (elt not in supervised_pred) & (elt in unsupervised_pred):
                        st.markdown("<mark style='background-color: lightgreen'>**" + elt + "**</mark>",
                                    unsafe_allow_html=True)
                    
                st.markdown("")
                st.markdown("<mark style='background-color: lightgray'>""</mark> &nbsp;Both models have predicted",
                            unsafe_allow_html=True)
                st.markdown("<mark style='background-color: lightblue'>""</mark> &nbsp;Only supervised model has predicted",
                            unsafe_allow_html=True)
                st.markdown("<mark style='background-color: lightgreen'>""</mark> &nbsp;Only unsupervised model has predicted",
                            unsafe_allow_html=True)
            else:
                st.markdown("#### Aucun tag prédit")
              
        else:
            st.info("Please, write your text before trying 'extraction tags'!")
    
    

    
if __name__ == '__main__':
        main()

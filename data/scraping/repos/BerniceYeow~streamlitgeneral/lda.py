# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:23:54 2020

@author: TNIKOLIC
"""
import streamlit as st
# components allow for pyLDAvis interactive graph display
import streamlit.components.v1 as components
import spacy
spacy.load("en_core_web_sm")
from spacy.lang.en import English
parser = English()
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
from gensim import corpora
import random
import pyLDAvis.gensim
import gensim
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
import math
from text_eda.helper_functions import open_html

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens   
 
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def main_function(input, NUM_TOPICS):
    
    text_data = []
    for i in input:
        tokens = prepare_text_for_lda(i)
        if random.random() > .99:
                #print(tokens)
                text_data.append(tokens)
                
    if len(text_data) == 0:
        tokens = prepare_text_for_lda(input[0])
        text_data.append(tokens)
        
    dictionary = corpora.Dictionary(text_data)
    
    corpus = [dictionary.doc2bow(text) for text in text_data]
    
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, 
                                               num_topics = NUM_TOPICS, 
                                               id2word=dictionary, 
                                               passes=10,
                                               random_state=100,
                                               eval_every=1,
                                               chunksize=100)
    
    topics = ldamodel.print_topics(num_words=4)
    
    for topic in topics:
        st.write(topic) 
        
    get_coherence(ldamodel,
                  corpus,
                  dictionary)
        
    ldamodel.save('model.gensim')

    # check if user input is 1 topic, if so, display error
    if NUM_TOPICS == 1:
        
        st.error("Please select more than 1 topic to visualise!")
        
    else:
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, 
                                                   corpus, 
                                                   dictionary, 
                                                   sort_topics=False,
                                                   mds='mmds')
        pyLDAvis.save_html(lda_display, 'lda_display.html')
        
        # display html page in streamlit
        # open file and decode it, then serve
        open_html("lda_display.html", height = 800, width = None)
        
def get_coherence(ldamodel,
                  corpus,
                  dictionary):
    
        
    coherence_model_lda = CoherenceModel(model=ldamodel, 
                                         corpus=corpus, 
                                         dictionary=dictionary, 
                                         coherence='u_mass')
    
    coherence_lda = coherence_model_lda.get_coherence()
    
    st.write("The u_mass coherence score for this model and number of topics is ", coherence_lda)
    
def main_function_optimal(input):
    
    text_data = []
    for i in input:
        tokens = prepare_text_for_lda(i)
        if random.random() > .99:
                #print(tokens)
                text_data.append(tokens)
                
    if len(text_data) == 0:
        tokens = prepare_text_for_lda(input[0])
        text_data.append(tokens)
                
    dictionary = corpora.Dictionary(text_data)
    
    corpus = [dictionary.doc2bow(text) for text in text_data]
        
    model_list, coherence_values, topic_list = compute_coherence_values(dictionary, 
                                                                         corpus, 
                                                                         limit=40, 
                                                                         start=2, 
                                                                         step=6)
    
    model_index = get_optimal_model(coherence_values)
    
    model_final = model_list[model_index]
    
    topic_final = topic_list[model_index]
        
    coh_score_final = coherence_values[model_index]
    
    st.write("The model with the optimal u_mass coherence score was trained on ",
             topic_final, " topics with a score of ", coh_score_final, ".",
             "Please see below plot for all coehrence scores and ",
             "corresponding topic numbers.")
    
    plot_best_topic_number(topic_final,
                           coh_score_final,
                           coherence_values)
    
    model_final.save('model.gensim')
    
    lda_display = pyLDAvis.gensim.prepare(model_final, 
                                        corpus, 
                                        dictionary, 
                                        sort_topics=False,
                                        mds='mmds')
    
    pyLDAvis.save_html(lda_display, 'lda_display.html')
        
    # display html page in streamlit
    # open file and decode it, then serve
    lda_display = open("lda_display.html", 'r', encoding='utf-8')
    source_code = lda_display.read() 
    components.html(source_code, height = 800, scrolling=True)   
    
    return(topic_final)
    
def compute_coherence_values(dic, corpus, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    # create a progress bar to display to user
    latest_iteration = st.empty()
    bar = st.progress(0)
    
    coherence_values = []
    model_list = []
    topic_list = []
    for num_topics in range(start, limit, step):
        
        # create progess percentages to pass to bar according on range
        value = num_topics / (limit - start)
        # update bar with every iteration
        latest_iteration.text(f'Training LDA on {num_topics} topics')
        bar.progress(value)
        
        model =  gensim.models.ldamodel.LdaModel(corpus, 
                                               num_topics = num_topics, 
                                               id2word=dic, 
                                               passes=10,
                                               random_state=100,
                                               eval_every=1,
                                               chunksize=100)
        model_list.append(model)
        topic_list.append(num_topics)
        coherence_model_lda = CoherenceModel(model=model, 
                                         corpus=corpus, 
                                         dictionary=dic, 
                                         coherence='u_mass')

        coherence_values.append(coherence_model_lda.get_coherence())

    return model_list, coherence_values, topic_list

def get_optimal_model(coherence_values):
    
    # get the minimum of u_mass scores
    best_coherence = min(coherence_values)
    
    # round the score down to nearest integer 
    best_coherence = math.ceil(best_coherence)
    
    # create a vector of rounded values
    coherence_values_1 = [math.ceil(num) for num in coherence_values]
    
    # find indexes where the values are = best_coherence
    index = [idx for idx, element in enumerate(coherence_values_1) if condition(element, best_coherence)]
    
    # create a vector of all coherence scores that match best_coherence
    best_coherence_final = []
    
    for i in index:
        best_coherence_1 = coherence_values[i]
        best_coherence_final.append(best_coherence_1)
        
    # take the first number in vector, shows where the model plateaus    
    best_coherence_final = best_coherence_final[0]
    
    # get the index of the model with best coherence score
    model_index = [idx for idx, element in enumerate(coherence_values) 
                   if condition(element, best_coherence_final)]
    
        
    model_index_final = model_index[0]
    
    return(model_index_final)

def condition(x, best_coherence): 
    return x == best_coherence

def plot_best_topic_number(topic_final,
                           coh_score_final,
                           coherence_values):
    
    limit=40; start=2; step=6;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.annotate("Optimum",
                 (topic_final,
                  coh_score_final))
    plt.plot(topic_final,
                  coh_score_final, 'go')
    st.pyplot()

def word_importance(topics):        
    warnings.filterwarnings("ignore")
    lda = gensim.models.LdaModel.load('model.gensim')
    fiz=plt.figure(figsize=(15,30))
    for i in range(topics):
        df=pd.DataFrame(lda.show_topic(i), columns=['Term','Prob']).set_index('Term')
        
        if topics < 5:
            plt.subplot(5,2,i+1)
            plt.tight_layout() 
        else:
            plt.subplot(topics,2,i+1)
            plt.tight_layout() 
            
        plt.title('Topic '+str(i+1))
        sns.barplot(x='Prob', 
                    y=df.index, 
                    data=df, 
                    label='Cities', 
                    palette='mako')
        
        plt.xlabel('Probability')
    
    st.pyplot()




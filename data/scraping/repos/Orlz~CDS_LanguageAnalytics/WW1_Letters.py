#!/usr/bin/env python

"""
===========================================================
Assignment 5: Topic Modelling with WW1 Letters 
===========================================================

This script performs topic modelling on a dataset of 50 letters between French or British soldiers and their loved ones during the First World War. 20 of the letters were originally in French and have been converted into English with the use of Google Translate (and some highschool French lessons! ;)) I hope you enjoy searching through the topics which these men and women chose to write about during the First World War. 

The script will work through the following steps: 
1. Load in and clean the data 
2. Generate bi-gram (a,b) and tri-gram (x, (a,b)) models using gensim 
3. Create a gensim dictionary and corpus
4. Build and run the LDA model 
5. Compute the Perplexity and coherence scores 
6. Creates a file of the topics and their most frequent words 
7. Visualise these topics into a html plot 

The script can be run from the command line by navigating to the correct directory and environment, then typing: 
    $ python3 WW1_Letters.py 

""" 

"""
------------------------
Import the Dependencies
------------------------
"""

#operating systems 
import os
import sys
sys.path.append(os.path.join(".."))
from pprint import pprint

# data handling 
import pandas as pd
import numpy as np 

#stopwords
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# nlp functionality 
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])


# visualisations 
import pyLDAvis.gensim
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 20,10

#LDA tools 
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import lda_utils

# warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

"""
---------------
Main Function 
---------------
"""

def main():
    """
    Here we'll call all the functions we want to run in our script (described below the main function) 
    """
    
    #Tell the user you're about to start up the process
    print("Hello, I'm setting up your WW1 letters topic modelling...") 
    
    # create the output directory
    if not os.path.exists("output"):
        os.mkdir("output")
        
    ##STEP 1: Load in the data (it has already been cleaned) 
    print("Loading in the data") 
    data = os.path.join("..", "data", "50_English_letters.csv")
    
    ##STEP TWO: Generate bi-gram and tri-gram models with gensim 
    print("I'm about to process the data and generate your bi and tri grams")
    data_processed = gensim_processing(data)
    
    
    ##STEP THREE: Create a gensim dictionary and corpus
    print("Models generated, now I'll create the dictionary and corpus")
    dictionary, corpus = create_dict_corpus(data_processed) 
    
    ##STEP FOUR: Run the LDA model (this will create 15 topics) 
    print("Set-up complete. Let's run the LDA model...") 
    lda_model = run_lda(corpus, dictionary)
    
    ##STEP FIVE: Calculate the perplexity and coherence scores 
    print("Calculating complexity and coherence...") 
    perplexity, coherence = calculate_plx_coh (data_processed, corpus, dictionary, lda_model) 
    
    ##STEP SIX: Create a file of the topics and their top words 
    print("Creating a txt file with the output topics. This will be found in output") 
    create_topics_df
    
    ##STEP SEVEN: Save the results as a simple txt file into output
    print("Creating a txt file with the perplexity and coherence scores. This will be found in output")
    save_results
    
    
    ##STEP SEVEN: Generate a topics plot and save it as a html 
    print("Creating a html plot which will be saved in output") 
    create_html_plot
    
    #Tell the user your script is finished 
    print("That's you finished, enjoy the results!")
    
    
"""
-----------
Functions 
-----------
"""       
    
def gensim_processing(data):
    """
    Here we use gensim to define bi-grams and tri-grams which enable us to create a create a dictonary and corpus 
    """
    #build the models first 
    bigram = gensim.models.Phrases(data["text"], min_count=3, threshold=100) #We're using a threshold of 100
    trigram = gensim.models.Phrases(bigram[data["text"]], threshold=100)  
    
    #Then fit them to the data 
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    #We further process the data using spacy and allow Nouns, Adjectives and Verbs to pass 
    data_processed = lda_utils.process_words(data["text"],nlp, bigram_mod, trigram_mod, allowed_postags=["NOUN","ADJ", "VERB"])

    #We now have a list of words which can be used to train the LDA model
    return data_processed
    
    
    
def create_dict_corpus(data_processed):
    """
    Here we create a dictonary and a corpus. 
    => The dictionary converts the words into an integer value
    => The corpus creates a 'bag of words' model for all the data (i.e. mixes it up and makes it unstructured) 
    
    """
    # Create Dictionary
    dictionary = corpora.Dictionary(data_processed)
    
    #We want to remove very common words so we'll filter those which appear in more than 80% of the letters
    #dictionary.filter_extremes(no_above=0.8)     (can be removed) 

    # Create Corpus: Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in data_processed]
    return dictionary, corpus 



def run_lda(dictionary, corpus):
    """
    Our model takes our data, corpus, and dictionary to generate a given number of topics. 
    This script uses 15 topics as that was the recommended number calculated. 
    
    """
    lda_model = gensim.models.LdaMulticore(corpus=corpus,            #our corpus 
                                           id2word=dictionary,       #our dictionary 
                                           num_topics=15,            # our number of topics defined as 15
                                           random_state=100,         #the number of random states (helps with repdoducability)
                                           chunksize=10,             #chunck size to help model be more effifienct 
                                           passes=10,                #Number of times the model passes through the data 
                                           iterations=100,
                                           per_word_topics=True,    
                                           minimum_probability=0.01)
    
    
    return lda_model

    
    
def calculate_plx_coh(data_processed, lda_model, corpus, dictionary):
    """
    Perplexity =  A measure of how good the model is. The lower the number the better. 
    Coherence =  
    
    """
    # Compute Perplexity
    perplexity = lda_model.log_perplexity(corpus)
    
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model= lda_model, 
                                     texts= data_processed, 
                                     dictionary= dictionary, 
                                     coherence= 'c_v')
    coherence = coherence_model_lda.get_coherence()
    
    print (f"\n The perplexity is {perplexity} and the coherence is {coherence}.") 
    
    
    return perplexity, coherence


     
def create_topics_df(lda_model, corpus, data_processed):
    
    """
    Here we look closer at the topics made and create a dataframe of these 
    
    """
    #print the topics to the terminal 
    pprint(lda_model.print_topics())
    
    
    #Create a data_frame of the topic keywords and save these as a csv 
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                          corpus=corpus, 
                                                          texts=data_processed)
    df_dominant_topic = df_topic_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    print(df_dominant_topic)
    
    df_dominant_topic.to_csv("output/topic_keywords.csv") 
    
    
    """
     We'll also look at the most dominent topics per letter by creating a matrix of topic values per letter
     Then we'll plot these into a lineplot using Seaborn
    
    """

    #Create a list of topics from your corpus 
    values = list(lda_model.get_document_topics(corpus))
    
    #Create an empty list called split 
    split = []
    
    #For every document in the corpus list(values) create an empty list called topic_prevelance
    for entry in values:
        topic_prevelance = []
        #For every topic in the document, add the contribution of this topic to each document into a column 
        for topic in entry:
            topic_prevelance.append(topic[1])
        #add this list with contributions to the empty split list created above 
        split.append(topic_prevelance)
        
        
    #Create the document-topic matrix and save it 
    df = pd.DataFrame(map(list,zip(*split)))
    df.to_csv("output/document_topic_matrix.csv")
    
    #Make this into a lineplot using Seaborn. We don't have many letters so our rolling mean will just be 5  
    topic_line_plot = sns.lineplot(data=df.T.rolling(5).mean())
    figure = topic_line_plot.get_figure()
    
    #Save the figure in the output directory 
    figure.savefig("output/topic_line_plot.png")
    print("\n Your topic line plot is saved in output.")
    
    

def save_results(perplexity, coherence):
    """
    We'll create a simple txt file to save our perplexity and coherence scores in. 
    """
    with open("output/perplexity_coherence.txt", "w+") as f:
        f.writelines(f"The models scores are as follows, \n\n Perplexity: {perplexity}, Coherence: {coherence}")
    
    

def create_html_plot(lda_model, corpus, dictionary):
    """ 
    Finally we'll create a html which allows the user to explore the topics themselves interactively
    """
    
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, f"output/LDA_vis.html")
    
    

if __name__=="__main__":
    #execute main function
    main()

#!/usr/bin/env python

"""
---------- Import libraries ----------
"""
# standard library
import sys,os
sys.path.append(os.path.join(".."))
from pprint import pprint

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# visualisation
import matplotlib.pyplot as plt
import pyLDAvis.gensim
import seaborn as sns
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 20,10


# LDA tools
import nltk
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils

"""
---------- Main function ----------
"""
def main():

    '''
    ------------------ Read data --------------------
    '''
    #Read in the data as a csv file
    filename = os.path.join("..", "data", "r_wallstreetbets_posts.csv")
    DATA = pd.read_csv(filename)
    
    '''
    The data set contains a lot of information which we do not need for our model. 
    This is information about username, individual links etc.
    We are primarily going to use the title column in the data set. This column contains the actual text.
    '''
    #Only use tree columns from the csv and a sample of 10000. 
    DATA = DATA[["title","created_utc", "score"]].sample(10000)
    
    #Split the text into individual senteces
    #Create empty list where the scenteces will be stored 
    output=[]
    #For every title in the column "title"
    print("Creating Doc object...")
    for title in DATA["title"]:
        #Create a doc object by using the spaCy NLP function
        doc = nlp(title)
        #Append to the list
        output.append(str(doc))

    '''
    ----------- Process using gensim and spaCy ----------------
    '''

    '''
    The next thing we do is using gensim to efficiently procude a model of bigrams and trigrams in the data.
    We first create bigrams based on words appearing one after another frequently. 
    These bigrams are then fed into a trigram generator, which takes the bigram as the second part of a bigram.
    '''

    # Build the bigram and trigram models
    print("Building bi- and trigrams...")
    bigram = gensim.models.Phrases(output, min_count=20, threshold=100) # a higher threshold gives fewer phrases.
    trigram = gensim.models.Phrases(bigram[output], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    '''
    We use the process_words function from our utils folder. 
    This function takes a text, nlp, bigram_mod, trigram_mod, stop_words and allowed_postags as arguments.
    It uses gensim to preprocess the words and uses spaCy to lemmatize and POS tag. 
    '''
    #Run the function with our arguments and set the allowed_postags to nouns and proper nouns
    print("Processing the data...")
    data_processed = lda_utils.process_words(output, nlp, bigram_mod, trigram_mod, allowed_postags=["NOUN", "PROPN"])

    #Create Dictionary
    #The dictionary converts each word into an integer value
    print("Creating Dictionary...")
    id2word = corpora.Dictionary(data_processed)

    # Create Corpus: Term Document Frequency. The corpus creates a 'bag of words' model for all of the data  
    print("Creating Corpus...")
    corpus = [id2word.doc2bow(text) for text in data_processed]
    
    
    '''
    --------------- Build LDA model ------------------------
    '''
    # Build LDA model using gensim
    print("Building LDA model...")
    lda_model = gensim.models.LdaMulticore(corpus=corpus,    # vectorised corpus - list of list of tuples
                                           id2word=id2word,  # gensim dictionary - mapping words to IDS
                                           num_topics=3,     # topics. This will be explained later in the script
                                           random_state=100, # set for reproducability
                                           chunksize=10,     # batch data for effeciency
                                           passes=10,        # number of full passes over data
                                           iterations=100,   # related to document rather than corpus
                                           per_word_topics=True, # define word distibutions
                                           minimum_probability=0.0) # minimum value
    
    '''
    -------------- Calculate model perplaxity ans coherence -------------------------
    '''

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
    #A measure of how good the model is. 
    #Calculate and return per-word likelihood bound, using a chunk of documents as evaluation corpus. 
    #It returns the variational bound score calculated for each word.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=data_processed, 
                                         dictionary=id2word, 
                                         coherence='c_v') #We use c_v as our choerence
    
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    
    '''
    -------------- Find the most optimal number of topics -------------------------
    '''
    
    '''
    We want to find the most optimal number of topics for our model. 
    Although the coherence value may be high at the high number of topics, it is not significant that it is the most optimal.
    One of the reasons for this is that there will be more repetitions of words the more topics there are. 
    So if one wants to avoid this, it may be an advantage with fewer topics.  
    '''
    print("Finding optimal topic number...")
    model_list, coherence_values = lda_utils.compute_coherence_values(texts=data_processed,
                                                                  corpus=corpus, 
                                                                  dictionary=id2word,  
                                                                  start=1, #The number of topics to start from
                                                                  limit=40, #The maximum number of topics 
                                                                  step=2) #The steps between the number of topics
    '''
    When we first ran the part to find the most optimal topic number, we got the number of 7 topics to be the most optimal. 
    But when we later in the script saw the visualization of how the topics are distributed,
    it became clear that they formed three main clusters, where the topics overlapped. 
    For this reason, we have chosen to include three topics in the model.
    '''
    
    '''
    --------------------- Find most dominant topic per chunk ---------------------
    '''
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                          corpus=corpus, 
                                                          texts=data_processed)
    #Reset the index
    df_dominant_topic = df_topic_keywords.reset_index()
    #Chose the columns for the dataframe
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.sample(10)
    
    #Display setting to show more characters in column
    pd.options.display.max_colwidth = 100

    #Create dataframe
    sent_topics_sorted_df = pd.DataFrame()
    #Use groupby on the column containing the dominant topic
    sent_topics_out_df_grpd = df_topic_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_out_df_grpd:
        #Concatenate the sent_topics_sorted_df with the column Perc_Contribution
        sent_topics_sorted_df = pd.concat([sent_topics_sorted_df, 
                                          grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                          axis=0)
    # Reset the index    
    sent_topics_sorted_df.reset_index(drop=True, inplace=True)

    #Choe the columns for the dataframe
    sent_topics_sorted_df.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

    '''
    --------------------- Create dataframe for the values ---------------------
    '''
    values = list(lda_model.get_document_topics(corpus))
    #Split tuples and keep only values per topic
    split = []
    for entry in values:
        topic_prevelance = []
        for topic in entry:
            topic_prevelance.append(topic[1])
        split.append(topic_prevelance)

    #Create document-topic matrix
    value_df = pd.DataFrame(map(list,zip(*split)))
    
    print("Saving output...")
    #Outpath for the dataframe
    df_outpath = os.path.join("..", "output", "value_df.csv")
    #Save datafram to a csv file
    value_df.to_csv(df_outpath)
    
    #Save vizualization to a png file
    sns.lineplot(data=value_df.T.rolling(50).mean())
    outpath_viz = os.path.join("..", "output", "topic_matrix_viz.png")
    plt.savefig(outpath_viz)
    print("Output saved")
    
#Define behaviour when called from command line
if __name__ == "__main__":
    main()    

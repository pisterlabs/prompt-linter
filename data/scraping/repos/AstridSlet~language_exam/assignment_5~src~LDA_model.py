#!/usr/bin/python

"""
Performing LDA topic modelling on twitter data, to explore topic prevalence over time.

Parameters:
    infile: str <path-to-input-data>
    threshold: int <threhshold-bigrams-trigrams>
    stepsize: int <stepsize-optimal-num-topics>
Usage:
    LDA_model.py <stepsize-optimal-num-topics> 2
Example:
    $ python LDA_model.py --stepsize 2
"""

# import dependencies 
import sys,os
from pprint import pprint
import argparse
sys.path.append(os.path.join(".."))

# data and nlp
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
import nltk
nltk.download('stopwords')

# visualisation
import pyLDAvis.gensim
import seaborn as sns
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 20,10


# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils_twitter

# warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


# define utility functions
def read_data(input_file):
    """
    Read data, make chunks of 10 tweets. 
    """
    # load file 
    data = pd.read_csv(input_file,
                       lineterminator='\n')
                       
    # make sure date col is datetime format
    data["date"] = pd.to_datetime(data["date"])
    # make data it is sorted by date
    data = data.sort_values(by="date")
    
    # make chunks of 10 tweets 
    tweet_chunks = []
    for i in range(0, len(data["tweet"]), 10):
        tweet_chunks.append(' '.join(data["tweet"][i:i+10]))
    return tweet_chunks 


def preprocess(chunks, t_value):
    """ 
    1. Build the bigram and trigram models
    2. Lemmatize and choose specific pos_tags and take only unigram, bigrams and trigrams above the threshold (t_value) = data_processed
    3. Create dictionary of the processed data (id2word)
    4. Create corpus: for each unigram/bigram/trigram count the freq (=Term Document Frequency)
    
    """
    # build models
    bigram = gensim.models.Phrases(chunks, min_count=3, threshold = t_value) 
    trigram = gensim.models.Phrases(bigram[chunks], threshold = t_value)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # lemmatize and use threshold
    data_processed = lda_utils_twitter.process_words(chunks,nlp,bigram_mod,trigram_mod,allowed_postags=["NOUN", "VERB", "ADJ"])        
    # create dictionary 
    id2word = corpora.Dictionary(data_processed)
    # create corpus
    corpus = [id2word.doc2bow(text) for text in data_processed]
    return data_processed, id2word, corpus


    
def optimal_topic(data_processed, corpus, id2word, step_size):
    """
    Try different number of topics (using stepsize) to find the optimal number.
    
    """
    # try topics
    model_list, coherence_values = lda_utils_twitter.compute_coherence_values(texts = data_processed,
                                                                              corpus = corpus, 
                                                                              dictionary = id2word,  
                                                                              start = 5, 
                                                                              limit = 25,  
                                                                              step = step_size)
    
    # get index of model with best coherence value 
    index = np.argmax(coherence_values)
    
    # get the optimal topic-number from this model
    optimal_num_topics = model_list[index].num_topics
    return optimal_num_topics


def run_model(data, corpus, id2word, optimal_num_topics):
    """
    Build LDA model and compute perplexity and coherence score.
    
    """
    # build model 
    lda_model = gensim.models.LdaMulticore(corpus = corpus,                 
                                           id2word = id2word,              
                                           num_topics = optimal_num_topics,
                                           random_state = 100,             
                                           chunksize = 10,                 
                                           passes = 10,                    
                                           iterations = 100,               
                                           per_word_topics = True,         
                                           minimum_probability = 0.0)       
    perplexity = lda_model.log_perplexity(corpus) 
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                     texts=data, 
                                     dictionary=id2word, 
                                     coherence='c_v')

    coherence = coherence_model_lda.get_coherence()
    return lda_model, perplexity, coherence


def save_output(lda_model, perplexity, coherence, topics):
    """
    Save model-information and topics in txt file.
    """
    f = open(os.path.join("..","output","final_output.txt"), 'w')
    f.write(f"The trained model has {topics} topics.\n")
    f.write(f"Perplexity score: {perplexity}.\n")
    f.write(f"Coherence score: {coherence}.\n")
    f.write("The topics of the model are: \n")
    sys.stdout = f
    pprint(lda_model.print_topics())
    f.close()

def create_viz(data_processed, lda_model, corpus):
    """
    Create visualization of topic prevalence over time.
    """
    # transform and prepare a LDA model’s data for visualization
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    values = list(lda_model.get_document_topics(corpus))
    split = []
    for entry in values:
        topic_prevelance = []
        for topic in entry:
            topic_prevelance.append(topic[1])
        split.append(topic_prevelance)
    # store info    
    df = pd.DataFrame(map(list,zip(*split)))
    plot = sns.lineplot(data=df.T.rolling(50).mean()).set_title("Topic prevalence over time")
    fig = plot.get_figure()    
    fig.savefig(os.path.join("..","output","topics_over_time.jpg"))

# define main function
def main(): 
    # initialise argumentparser
    ap = argparse.ArgumentParser()
    # define arguments
    ap.add_argument("-i", 
                    "--infile", 
                    required = False, 
                    type     = str, 
                    default  = "prepro_subset.csv", 
                    help="Input filename")

    ap.add_argument("-t", 
                    "--threshold", 
                    required = False, 
                    type     = int,
                    default  = 50,
                    help     = "Threshold value for the bigram and trigram models")
    
    ap.add_argument("-s", 
                    "--stepsize", 
                    required = False, 
                    type     = int,
                    default  = 5,
                    help     = "Stepsize when determining the optimal number of topics for the model")


    # parse arguments to args
    args = vars(ap.parse_args())
    
    # get input filepath
    input_file = os.path.join("..", "data", args["infile"])
    
    # read data
    twitter_chunks = read_data(input_file)
        
    # get threshold 
    t_value = args["threshold"]
    
    print("[INFO] Building the bigram and trigram models...")
    # build the bigram and trigram models 
    data_processed, id2word, corpus = preprocess(twitter_chunks, t_value)
    
    print("[INFO] Finding optimal number of topics...")
    # get stepsize 
    step_size = args["stepsize"]
    # find optimal number of topics
    optimal_num_topics = optimal_topic(data_processed, 
                                       corpus, 
                                       id2word, 
                                       step_size)
    print(f"[RESULT] The optimal number of topics for this model is {optimal_num_topics}\n")    

    # build LDA model 
    lda_model, perplexity, coherence = run_model(data = data_processed, corpus = corpus,id2word = id2word,optimal_num_topics = optimal_num_topics)
    
    
    # create visualization
    create_viz(data_processed, lda_model, corpus)
    
    # save model
    save_output(lda_model, perplexity, coherence, topics=optimal_num_topics)
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()
    
    
    
    
    
    

#!/usr/bin/env python
"""
Specify file path of the csv file of Trump tweets and name of the output graph. You can also specify number of topics and what kind of words you want to investigate. The default is 10 topics and word types nouns and verbs. The output will be a perplexity and coherence score printed in the terminal as well as a print of the 10 most prominent words constituting each topic. Furthermore, a plot of the development of topics within Trumps tweets will be saved in a folder called out in the path relative to the working directory i.e., location of the script.
Parameters:
    input_file: str <filepath-of-csv-file>
    output_filename: str <name-of-png-file>
    n_topics: int <number-of-topics>
    word_types: list <list-of-word-types>
Usage:
    development_of_trump.py -f <filepath-of-csv-file> -o <name-of-png-file> -n <number-of-topics> -w <list-of-word-types>
Example:
    $ python3 development_of_trump.py -f ../data/Trump_tweets.csv -o trumps_development.png -n 15 -w "['NOUN', 'VERB']"
    
## Task
- Train an unsupervised classifyer as an LDA model on your data to extract structured information that can provide insight into topics in Trumps tweets.
- Output is in the form of an html file containing the topics (as well as a print in the terminal) and a png file for the development of topics. Both can be found in the folder data/output.
"""

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# standard library
import sys,os
#sys.path.append(os.getcwd())
sys.path.append(os.path.join(".."))
from pprint import pprint

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

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
from utils import lda_utils

# warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import argparse

# argparse 
ap = argparse.ArgumentParser()
# adding argument
ap.add_argument("-f", "--input_file", required = True, help= "Path to the csv-file")
ap.add_argument("-o", "--output_filename", default = "trumps_development.png", help = "Name of output file")
ap.add_argument("-n", "--n_topics", default = 10, help = "Number of topics")
ap.add_argument("-w", "--word_type", default = "['NOUN', 'VERB']", help = "Type of word. Choose between: 'NOUN','VERB','ADJ','ADV'")
# parsing arguments
args = vars(ap.parse_args())




def main(args):
    # get path to the csv file
    in_file = args["input_file"]
    # name for output file
    out_file = args["output_filename"]
    # number of topics
    n_topics = int(args["n_topics"])
    # word types
    word_types = args["word_type"]
    
    
    # Initialize class object
    Trump = Tweet_development(input_file = in_file, output_file = out_file, n_topics = n_topics, word_types = word_types)
    
    # use process_data method and save returned dataframe
    data_processed = Trump.process_data()
    
    # build the lda model
    id2word, corpus, lda_model = Trump.lda_model(data_processed)
    
    # plot development of tweets
    Trump.development_and_outputs(id2word = id2word, corpus = corpus, lda_model = lda_model)
    
    # print done
    print("Good job! The script is now done. Have a nice day!")
    
class Tweet_development:
    def __init__(self, input_file, output_file, n_topics, word_types):
        '''
        Constructing the Tweet_development object
        '''
        # creating the class object with the user defined inputs
        self.input_file = input_file
        self.output_file = output_file
        self.n_topics = n_topics
        self.word_types = word_types
        
    def load_and_prepare(self):
        '''
        Loading the input data. Filter and prepare data for classification.
        Returns a filtered list.
        '''
        print("\nLoading the data...")
        # read csv file
        df = pd.read_csv(self.input_file)
        
        # remove hyperlinks by filtering
        filtering = df['text'].str.contains("http*")
        df = df[~filtering]
       
        # remove retweets
        data = df[df["isRetweet"]== "f"]
        
        # make a corpus of the contents column only
        tweets = data['text'].values.tolist()
        
        return tweets

    def process_data(self):
        '''
        Building bigram and trigram models and fitting them to the data.
        Reducing the feature space by lemmatizing and POS tagging the corpus of tweets.
        Returns the processed data.
        '''
        
        tweets = self.load_and_prepare()
        
        print("\nBuilding bigram and trigram models and fitting it to the data")
        
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(tweets, min_count=3, threshold=75) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[tweets], threshold=75)  

        # fitting the models to the data
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        
        print("\nLemmatizing the data and doing POS tagging. This takes a few minutes...")
        # Processing the data using Ross' lda utils function
        data_processed = lda_utils.process_words(tweets,
                                                 nlp, 
                                                 bigram_mod,
                                                 trigram_mod,
                                                 allowed_postags=self.word_types)
        return data_processed
    
    
    def lda_model(self, data_processed):
        '''
        Creating dictionary and corpus of word frequency from the processed data.
        Building and evaluating the LDA model.
        Print perplexity scores and coherence scores.
        Print the top ten words representing each topic.
        '''       
        
        # Create Dictionary
        id2word = corpora.Dictionary(data_processed)

        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_processed]
        
        print(f"Building the LDA model. You have {len(data_processed)} tweets in the data so it might take a while")
        # Build LDA model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,             # vectorised corpus - list of lists of tupols
                                               id2word=id2word,           # gensim dictionary - mapping words to IDs
                                               num_topics=self.n_topics,  # number of topics set by user or default
                                               random_state=100,          # random state for reproducibility
                                               chunksize=30,              # batch data for efficiency
                                               passes=10,                 # number of times to pass over the data set to create better model
                                               iterations=100,            # related to document rather than corpus
                                               per_word_topics=True,      # define word distributions 
                                               minimum_probability=0.0)   # minimum value
        
        # Compute Perplexity and print in the terminal
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, 
                                     texts=data_processed, 
                                     dictionary=id2word, 
                                     coherence='c_v')
        
        # get the coherence score and print in the terminal
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        
        # print the topics found by the lda model in the terminal
        pprint(lda_model.print_topics())
        
        return id2word, corpus, lda_model
    
    
    def development_and_outputs(self, id2word, corpus, lda_model):
        '''
        Calculate dominant topic for each document/tweet and plot development of topics over time with seaborn. Save plot as png.
        '''
        print("\nCreating visualizations and saving as html and png in output folder")
        # Create output directory if it doesn't exist
        outputDir = os.path.join("..", "data", "output")
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
            print("Directory " , outputDir ,  " Created ")
        else:
            print("Directory " , outputDir ,  " already exists")
        
        # Make gensim LDA visualization
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
        # define output path and filename
        out_path = os.path.join(outputDir, "lda_topics.html")
        # save visualization as html
        pyLDAvis.save_html(vis, out_path)
        print("\nGensim LDA visualization is saved as", out_path)
        
        
        # inspect dominant topic
        values = list(lda_model.get_document_topics(corpus))
        
        # create empty list 
        split = []
        # for loop for each document in the corpus
        for entry in values:
            # create empty list
            topic_prevelance = []
            # for loop for each topic in each document
            for topic in entry:
                # append the contribution of each topic for each document
                topic_prevelance.append(topic[1])
            # append the list with contributions of topics to the split list    
            split.append(topic_prevelance)
        
        # making a dataframe containing for each document the percentage of contribution of the 10 topics
        df = pd.DataFrame(map(list,zip(*split)))
        
        # defining the output path
        out_path = os.path.join(outputDir, self.output_file)
        
        # making a lineplot with a rolling mean of 500 tweets
        line_plot = sns.lineplot(data=df.T.rolling(500).mean())
        # saving the lineplot as a figure
        fig = line_plot.get_figure()
        # saving the figure in the output path
        fig.savefig(out_path)
        print("\nLineplot for development of topics in tweets is saved as", out_path)
        
        
if __name__ == "__main__":
    main(args)
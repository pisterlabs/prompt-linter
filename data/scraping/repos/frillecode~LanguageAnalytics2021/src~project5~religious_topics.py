# Path and system operations
import sys,os
sys.path.append(os.path.join("..", ".."))
import pandas as pd
import random
from collections import Counter
import argparse 

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Regex tools
import re
import string

# Spacy
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load(disable=["ner"])

# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils
from gensim.test.utils import datapath

# Visualization 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class ReligiousTP:
    def __init__(self, args):
        self.args = args
        
    def read_txt(self):
        '''
        This functions takes a filename or list of filenames, loads it/them and returns the data.
        '''
        with open(os.path.join("..", "..", "data", "project5",  self.args['filename']), 'r', encoding="utf-8") as f:
            book = f.read()
            
        self.basename = os.path.splitext(os.path.basename(self.args['filename']))[0]

        return book
    
    # Preprocessing
    def remove_things(self, book):
        print("[INFO] Preprocessing text...")
        ''' Preprocessing function to remove digits, newlines, most punctuation. The data is filtered to only contain the content of the books (and not the introduction from Gutenberg). In addition, the function ensures that the len of the text does not exceed the max for spacy (1,000,000 characters)
        Input:
            book: str, text data in one long string
        Output:
            text: str, preprocessed text data
        '''
        # Filters Gutenberg related text
        if self.basename == "pg2800": #intro text different for pg2800.txt
            text = book[book.find("END THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*Ver.04.29.93*END"):book.find("End of The Project Gutenberg Etext of The Koran")] 
        else: 
            text = book[book.find("* START OF THIS PROJECT GUTENBERG EBOOK"):book.find("End of the Project Gutenberg EBook of")]

        # Data cleaning
        text=re.sub(r'[^\w.\s]', '', text) #remove everything except word characters, space, and '.'
        text=re.sub(r'\s+', ' ', text) #remove newline without removing spaces
        text=re.sub(r'\d+', '', text) #remove digits

        # Ensure that len of text does not exceed spacy's limit
        if len(text) > 1000000:
            text = text[:100000]

        return text
    
    def make_corpus(self, text, nlp):
        '''This function takes a text and Language object, splits the text into chunks and performs different preprocessing steps - including stopword removal, forming bigrams and trigrams, lemmatization and pos-tagging. 
        Lastly, a dictionary with an integer value for each word and a corpus with 'bag of words' model for all chunks are returned.
        Input:
            text: str, text data
            nlp: Language object (e.g. nlp = en_core_web_sm.load(disable=["ner"]))
        Output:
            id2word: dict, dictionary with an integer value for each word
            corpus: list of (token_id, token_count) 2-tuples, Term Document Frequency within chunks (bag-of-words)
        '''
        print("[INFO] Creating text corpus...")
       
        # Doc object
        doc = nlp(text)

        # Split into chunks
        sentences = [sent.string.strip() for sent in doc.sents] #sentences

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(sentences, min_count=3, threshold=100)
        trigram = gensim.models.Phrases(bigram[sentences], threshold=100)  

        # Fitting the models to the data
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # Perform additional preprocessing steps (stopwords, lemmas, pos)
        text_processed = lda_utils.process_words(sentences, nlp, bigram_mod, trigram_mod, allowed_postags=['NOUN', "ADJ"])

        # Create Dictionary
        id2word = corpora.Dictionary(text_processed)
        
        id2word.filter_extremes(no_below=5, no_above=0.5)

        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in text_processed]
        
        return text_processed, id2word, corpus
    
    def compute_metrics(self, dictionary, corpus, texts, limit=30, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Input:
            dictionary : Gensim dictionary
            corpus : Gensim corpus
            texts : List of input texts
            limit : Max num of topics

        Output:
            model_list : List of LDA topic models
            coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
    
        print("[INFO] Calculating coherence and perplexity scores...")


        random.seed(2021)
        coherence_values = []
        perplexity = []
        model_list = []

        # Computing LDA model for each n of topics 
        for topic_n in range(start, limit, step):
            model = gensim.models.LdaMulticore(corpus=corpus, 
                                               num_topics=topic_n, 
                                               id2word=dictionary) 
            model_list.append(model)

            # Perplexity and coherence values
            perplexity.append(model.log_perplexity(corpus))

            coherencemodel = CoherenceModel(model=model, # Pre-trained topic model
                                            texts=texts, # Tokenized texts
                                            dictionary=dictionary, # Gensim dictionary mapping of id word to create corpus
                                            coherence='c_v') # Coherence measure to be used 

            coherence_values.append(coherencemodel.get_coherence())


        # Plot and save
        x = range(start, limit, step)
        if self.args['metric'] == 'coherence':
            plt.plot(x, coherence_values)
        elif self.args['metric'] == 'perplexity':
            plt.plot(x, perplexity)
        plt.xlabel("Number of topics")
        plt.ylabel(f"{self.args['metric']} score")
        plt.legend((f"{self.args['metric']}_values"), loc='best')
        plt.savefig(os.path.join(self.args['outpath'], f"{self.args['metric']}.png"))
        plt.show()

        # Dataframe with metric
        info = pd.DataFrame(zip(x, coherence_values, perplexity), columns = ["n_topic", "coherence", "perplexity"]) 
        if self.args['metric'] == 'coherence':
            # Print the coherence scores
            info = info.sort_values("coherence", ascending = False, ignore_index = True)    
            self.optimal_topics = info.n_topic.loc[0]
        elif self.args['metric'] == 'perplexity':
            info = info.sort_values("perplexity", ascending = True, ignore_index = True)    
            self.optimal_topics = info.n_topic.loc[0]
        # Using .loc we can index the df and get the zero row and print this using formatted strings
        print(f"Topic with highest {self.args['metric']} score is topic number {self.optimal_topics}")

        return model_list, info

    def LDA(self, corpus, id2word, text_processed):
        '''
        Compute LDA model using gensim.

        Input:
            corpus : Gensim corpus
            id2word: dict, dictionary with an integer value for each word
            text_processed : List of input texts
            num_topics: Num of topics

        Output:
            lda_model : LDA topic model
        '''        
        print("[INFO] Building LDA model...")

        # Check whether metric has been specified in the command-line. Otherwise, use num_topics value (default or from arg)
        if self.args['metric'] is not None:
            self.num_topics = self.optimal_topics
        else:
            self.num_topics = self.args['num_topics']

        # Build LDA model
        self.lda_model = gensim.models.LdaMulticore(corpus=corpus,      #vectorized corpus - list of lists of tuples
                                       id2word=id2word,                 #gensim dict - mapping word to IDS
                                       num_topics=self.num_topics,      #number of topics
                                       random_state=100,                #set for reproducibility
                                       eta = "auto",                    #prior on the per-topic word distribution
                                       alpha = "asymmetric",            #prior on the per-document topic distributions
                                       chunksize=10,                    #batch data for efficiency
                                       passes=10,                       #number of full passes over data 
                                       iterations=100,                  #number of times going over single document 
                                       per_word_topics=True,            #define word distributions
                                       minimum_probability=0.0)         #minimum value 

        # Evaluate final model        
        print('\nPerplexity: ', self.lda_model.log_perplexity(corpus)) #perplexity
        
        coherence_model_lda = CoherenceModel(model=self.lda_model, #coherence  
                                     texts=text_processed,
                                     dictionary=id2word,
                                     coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
 
            
    
    def get_topics(self, corpus, text_processed):
        '''
        Prints most dominant topics calculated by the topic-model.

        Input:
            lda_model : topic-model
            text_processed : List of input texts
            corpus: Gensim corpus            
        '''
        print("[INFO] Retrieving most dominant topics...")

        # Display setting to show more characters in column
        pd.options.display.max_colwidth = 100

        # Results
        topic_keywords = lda_utils.format_topics_sentences(ldamodel=self.lda_model,
                                                              corpus=corpus,
                                                              texts=text_processed)

        topics_sorted = pd.DataFrame()
        topics_outdf_grpd = topic_keywords.groupby('Dominant_Topic')

        for i, grp in topics_outdf_grpd:
            topics_sorted = pd.concat([topics_sorted,
                                              grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                              axis=0)

        # Reset Index    
        topics_sorted.reset_index(drop=True, inplace=True)

        # Format
        topics_sorted.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

        # Save
        topics_sorted.to_csv(os.path.join(self.args['outpath'],f"{self.basename}_topics_sorted.csv"))
        
                             
                             
    def plot_results(self, text_processed):
        '''
        Generates and saves plot of word weight and occurrences in each topic
        '''
        if self.num_topics < 8: #only plot is number of topics is below 8 (for simplicity of overview)
            # Collect topics
            topics = self.lda_model.show_topics(formatted=False)
            data_flat = [w for w_list in text_processed for w in w_list]
            counter = Counter(data_flat)
            
            out = []

            # Count occurrence of word and its weight in topic
            for i, topic in topics:
                for word, weight in topic:
                    out.append([word, i , weight, counter[word]])
                    
            df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

            # Plot
            fig, axes = plt.subplots(self.num_topics, 1, figsize=(16,20), sharey=True, dpi=160)
            
            # Colors
            cols = [color for name, color in mcolors.BASE_COLORS.items()]
            
            # Take every subplot 
            for i, ax in enumerate(axes.flatten()):
                ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
                ax_twin = ax.twinx()
                ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
                ax.set_ylabel('Word Count', color=cols[i])
                ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
                ax.tick_params(axis='y', left=False)
                ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
                ax.legend(bbox_to_anchor=(1, 0.9)); ax_twin.legend(loc='upper right')
            
            # Define layout
            fig.tight_layout(w_pad=2)    
            fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
            plt.savefig(os.path.join(self.args['outpath'],f"{self.basename}_word_weight_occurrence.png")) #save figure
            plt.show()
            
            print(f"[INFO] Figure with topics and keywords is saved in '{self.args['outpath']}/' as '{self.basename}_word_weight_occurrence.png'")
                                
                                
def main():
    ap = argparse.ArgumentParser(description="[INFO] class made to perform topic modelling on religious texts") 
    ap.add_argument("-f", 
                "--filename", 
                required=False, 
                type=str, 
                default= "pg10.txt", # Bible
                help="str, filename for txt file") 
                             
    ap.add_argument("-o", 
                "--outpath", 
                required=False, 
                type=str, 
                default= os.path.join("out"), 
                help="str, folder for output files")  
    
    ap.add_argument("-m", 
                "--metric", 
                required=False, 
                choices = ["coherence", "perplexity"],
                type=str, 
                help="str, method to approximate number of topics with")
                             
    ap.add_argument("-n", 
                "--num_topics", 
                required=False, 
                default=5,
                type=int, 
                help="int or none, number of topics to model")                                                  
                             
    args = vars(ap.parse_args())

    ReligiousTopic = ReligiousTP(args=args)
    
    book = ReligiousTopic.read_txt()
    
    filtered_text = ReligiousTopic.remove_things(book)
    
    text_processed, id2word, corpus = ReligiousTopic.make_corpus(filtered_text, nlp)
    
    if args['metric'] is not None:
        models, info = ReligiousTopic.compute_metrics(id2word, corpus, text_processed)
    
    ReligiousTopic.LDA(corpus, id2word, text_processed)
    
    ReligiousTopic.get_topics(corpus, text_processed)
    
    ReligiousTopic.plot_results(text_processed)
                             
if __name__=="__main__":
    main()      
    print("[INFO] DONE! ")    

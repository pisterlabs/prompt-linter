import os
import json
import datetime
import csv
import string
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

import nltk
from nltk.corpus import words, stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import collections

import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#dictionary_words = dict.fromkeys(words.words(), None)
   
#stopWords = set(stopwords.words('english'))
#tokenizer = RegexpTokenizer(r'\w+')

#stemmer = PorterStemmer()
#lemmatiser = WordNetLemmatizer()

stop = set(stopwords.words('english'))
stop.add ('u')
stop.add ('e')
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

#topic_doc_complete = []
#lda_model = ""


class TwitterTopics:   

    def __init__(self, folder_path, mongoDB_database=None):
                    
        self.folder_path = folder_path        
        self.lda_model = object()
        self.lsi_model = object()
        self.doc_term_matrix = object()
        self.dictionary = object()
        self.lda_coh_u_mass = 0
        self.lda_coh_c_v = 0
        self.lsi_coh_u_mass = 0
        self.lsi_coh_c_v = 0          
        
        self.db = mongoDB_database
        if mongoDB_database is not None:
            self.c_topics = self.db.topics
        else:
            self.c_topics = None
                
        
    def __del__(self):
        self.folder_path = None        
        self.lda_model = None
        self.lsi_model = None
        self.doc_term_matrix = None
        self.dictionary = None
        self.lda_coh_u_mass = None
        self.lda_coh_c_v = None
        self.lsi_coh_u_mass = None
        self.lsi_coh_c_v = None
                

    def get_coh_u_mass(self):
        return self.lda_coh_u_mass, self.lsi_coh_u_mass
    
    def get_coh_c_v(self):
        return self.lda_coh_c_v, self.lda_coh_c_v
    

    #create one array with all tweets of one hashtag for topic analysis
    def get_docs_from_file(self, file_path):

        docs = []
        
        with open(file_path, 'r', encoding='utf8', errors='ignore') as f:                            
            for line in f:        
                docs.append(line)

        f.close()
        return docs


    #clean documents for topic analysis
    def clean_docs(self, doc, delete_numbers=True, delete_stop_words=True, lemmatize_words=True): 
        
        doc_clean = doc
        
        if delete_numbers ==True:
            doc_clean = doc.replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace('0', '')
            
        if delete_stop_words == True:
            doc_clean = " ".join([i for i in doc_clean.lower().split() if i not in stop])
        
        doc_clean = ''.join(ch for ch in doc_clean if ch not in exclude)
        
        if lemmatize_words == True:
            doc_clean = " ".join(lemma.lemmatize(word) for word in doc_clean.split())

        return doc_clean
        

    #train model
    def train_model(self, topic_docs, num_topics, model_name, blnSaveinDB=False, blnSaveTrainedModelFiles=False, txtFileName=None,
                    model_type='both', lda_num_of_iterations=150, delete_stop_words=True, lemmatize_words=True, delete_numbers=True):
        
        #starttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #print("Executing train_model... Started at: " + starttime )        

        doc_clean = [self.clean_docs(doc, delete_numbers, delete_stop_words, lemmatize_words).split() for doc in topic_docs]

        # Creating the term dictionary of our corpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
        self.dictionary = corpora.Dictionary(doc_clean)

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        self.doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in doc_clean]

        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        
        file_data = []        
        
        if model_type in ('lda', 'both'):
            # Build the LDA model
            self.lda_model = gensim.models.LdaModel(corpus=self.doc_term_matrix, num_topics=num_topics, id2word=self.dictionary, iterations=lda_num_of_iterations)                            
            #get LDA coherence
            self.lda_coh_u_mass = CoherenceModel(model=self.lda_model, corpus=self.doc_term_matrix, dictionary=self.dictionary, coherence='u_mass') 
            self.lda_coh_c_v = CoherenceModel(model=self.lda_model, texts=doc_clean, dictionary=self.dictionary, coherence='c_v')
            
            #create json file with lda results
            for idx in range(num_topics):                
                topic = idx+1
                strtopic = str(topic)
                data = '{"model_name":"' + model_name + \
                        '", "model_type":"' + 'lda' + \
                        '", "timestamp":"' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + \
                        '", "no_tweets":"' + str(len(topic_docs)) + \
                        '", "coh_u_mass":"' + str(self.lda_coh_u_mass.get_coherence()) + \
                        '", "coh_c_v":"' + str(self.lda_coh_c_v.get_coherence()) + \
                        '", "topic_no":"' + strtopic + \
                        '", "topic":"' + str(self.lda_model.print_topic(idx, num_topics)).replace('"', "-") + '"}'
                x = json.loads(data)
                file_data.append(x)
            
                
        if model_type in ('lsi', 'both'):
            # Build the LSI model
            self.lsi_model = gensim.models.LsiModel(corpus=self.doc_term_matrix, num_topics=num_topics, id2word=self.dictionary)    
            #get LSI coherence
            self.lsi_coh_u_mass = CoherenceModel(model=self.lsi_model, corpus=self.doc_term_matrix, dictionary=self.dictionary, coherence='u_mass') 
            self.lsi_coh_c_v = CoherenceModel(model=self.lsi_model, texts=doc_clean, dictionary=self.dictionary, coherence='c_v')
        
            #create json file with lsi results
            for idx in range(num_topics):
                topic = idx+1
                strtopic = str(topic)
                data = '{"model_name":"' + model_name + \
                        '", "model_type":"' + 'lsi' + \
                        '", "timestamp":"' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + \
                        '", "no_tweets":"' + str(len(topic_docs)) + \
                        '", "coh_u_mass":"' + str(self.lsi_coh_u_mass.get_coherence()) + \
                        '", "coh_c_v":"' + str(self.lsi_coh_c_v.get_coherence()) + \
                        '", "topic_no":"' + strtopic + \
                        '", "topic":"' + str(self.lsi_model.print_topic(idx, num_topics)).replace('"', "-") + '"}'
                x = json.loads(data)
                file_data.append(x)


        # Save if mongoDB collection is asked
        if blnSaveinDB == True:        
            if self.db  is not None:
                self.c_topics.insert_many(file_data)                
            else:
                print("Can't save topics in db. No mongoDB connection was set up.")
                    
        # Save results in a text file
        if txtFileName is not None:
            with open(txtFileName, 'w', encoding="utf-8") as outfile:
                json.dump(file_data, outfile)
    

            
            
        # Save models into file
        if blnSaveTrainedModelFiles == True:
            
            #creates path if does not exists
            if not os.path.exists(self.folder_path + "/trained_models/"):
                os.makedirs(self.folder_path + "/trained_models/")
            
            self.lda_model.save(self.folder_path + "/trained_models/" + model_name + "_lda_model.model")
            self.dictionary.save(self.folder_path + "/trained_models/" + model_name + "_dictionary.dict")
        
        
        #endtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #print("Finished executing train_model. Ended at: " + endtime)
    

    #train model from file
    def train_model_from_file(self, file_path, num_topics, model_name, blnSaveinDB=False, blnSaveTrainedModelFiles=False, txtFileName=None,
                    model_type='both', lda_num_of_iterations=150, delete_stop_words=True, lemmatize_words=True, delete_numbers=True):        
        docs = self.get_docs_from_file(file_path)
        self.train_model(docs, num_topics, model_name, blnSaveinDB, blnSaveTrainedModelFiles, txtFileName, model_type, lda_num_of_iterations, delete_stop_words, lemmatize_words, delete_numbers)        
    
    
    
        
    
    #plot graph with lda topics
    def plot_topics(self, file_name, no_of_topics, model_type = 'lda', fig_size_x = 17, fig_size_y=15, replace_existing_file=True):
        
        
        if replace_existing_file==True or not os.path.exists(file_name):
                
            fig_size_y = 7*(no_of_topics/2)        
            fiz=plt.figure(figsize=(fig_size_x, fig_size_y))

            for i in range(no_of_topics):
                if model_type == 'lda':
                    df=pd.DataFrame(self.lda_model.show_topic(i), columns=['term','prob']).set_index('term')        
                elif model_type == 'lsi':
                    df=pd.DataFrame(self.lsi_model.show_topic(i), columns=['term','prob']).set_index('term')        

                no_rows = int(no_of_topics/2)+no_of_topics%2            
                plt.subplot(no_rows,2,i+1)
                plt.title('topic '+str(i+1))
                sns.barplot(x='prob', y=df.index, data=df, label='Cities', palette='Reds_d')
                plt.xlabel('probability')

            #save the file 
            plt.savefig(file_name, dpi=200, facecolor='w', edgecolor='w')

            #plt.show()
            plt.cla()   # Clear axis
            plt.clf()   # Clear figure
            plt.close() # Close a figure window
                

    
    # read a frequency list into a pandas objects
    # file format word\tfrequency
    def read_freq_list_file(self, file_path, delimiter='\t'):
        #df = pd.read_csv(file_path, encoding = "ISO-8859-1", header=None, sep=delimiter, lineterminator='\n')        
        df = pd.read_csv(file_path, encoding = "utf-8", header=None, sep=delimiter, lineterminator='\n')
        
        df.columns = ['word', 'freq']
        
        return df
    
    
    #plot a bar graph with the top frequency list
    def plot_top_freq_list(self, fr_list, top_no, ylabel, exclude_top_no=0, file=None, replace_existing_file= True):
                        
        if exclude_top_no != 0:
            fr_list = fr_list.iloc[exclude_top_no:]
        
        fr_list = fr_list.nlargest(top_no,'freq')                            
        
        
        if len(fr_list) < top_no:
            for i in range( int((top_no-len(fr_list)) / 2.5)):
                
                data = [['', 0], ['', 0] ]    
                df2 = pd.DataFrame(data, columns = ['word', 'freq'])  
                fr_list = fr_list.append(df2)            

        
        fr_list_gr = fr_list.groupby("word")
            
        plt.figure(figsize=(12, len(fr_list)/2.5))                
        fr_list_gr.max().sort_values(by="freq",ascending=True)["freq"].plot.barh()
        plt.xticks(rotation=50)
        plt.xlabel("Frequency")
        plt.ylabel(ylabel)
        if file != None:
            if replace_existing_file==True or not os.path.exists(file):
                plt.savefig(file, dpi=300, bbox_inches='tight')
                
        #plt.show()
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window
                
        
    
    #plot a word cloudfor a frequency list
    def plot_word_cloud(self, fr_list, file=None, replace_existing_file=True):                
        
        wordcount = collections.defaultdict(int)

        for index, row in fr_list.iterrows():            
            wordcount[row['word']] = row['freq']
    
        try:
            wordcloud = WordCloud(width=2000, height=1300, max_words=1000, background_color="white").generate_from_frequencies(wordcount)
        except:
            wordcloud = WordCloud(width=2000, height=1300, background_color="white").generate_from_frequencies(wordcount)            

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        
        if file is not None:
            if replace_existing_file==True or not os.path.exists(file):
                plt.savefig(str(file), dpi=300)
                        
        #plt.show()            
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window
                        

    
    #load existing model from file
    #predict topic of a new tweet based on model
    
    
    
    
    

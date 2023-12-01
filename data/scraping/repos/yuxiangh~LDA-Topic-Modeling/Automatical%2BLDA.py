
# coding: utf-8

# In[1]:


from nltk.corpus import stopwords
import string
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import spacy
get_ipython().magic(u'matplotlib inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[2]:


class TopicModeling(object):
    def __init__(self,datasets,penality):
        self.datasets=datasets
        
        self.penality=penality
        self.sentences=datasets["Sentence"]
        self.prepocessed_words=[]
        
        self.number_topics=None
        self.avg_cosine_similarity=None
        self.coherence_score=None
        self.perplexity=None
        self.Best_topic_numbers=None
        self.bigram_mod=None
        self.trigram_mod=None
        self.data_words_bigrams=None
        self.data_words_trigrams=None
        self.data_lemmatized=None
        self.id2word=None
        self.corpus=None
        self.LDA_model=None
        self.topics_number=None
        self.labels=None
        self.plot=None
    
    #run prepocessing part, and then run the LDA_Modeling part
    def Run_Prepocessing(self):
        self.text_process()
        self.build_gram_model()
        self.make_bigrams()
        self.make_trigrams()
        self.lemmatization()
        self.build_Id2Word_Corpus()
    
        
    def Run_LDA_Model(self,number_topics):
        self.number_topics=number_topics
        self.build_LDA_model()
        self.visualize_LDA_model()
        self.getAvgCosineSimilarity()
        self.getTopicFiveKeyWord()
        self.getEachSentenceLabel()
    
    
    def select_Best_Number_Topics(self):
        best=1
        best_number_topics=0
        for i in range(0,len(self.topics_list)):
            number_topics=self.topics_list[i]
            avg_cosine_similarity=self.avg_cosine_similarity_list[i]
            if avg_cosine_similarity<best:
                best=avg_cosine_similarity
                best_number_topics=number_topics
        self.Best_topic_numbers=best_number_topics
        
    
    def MakePlot(self):
        X=self.topics_list
        Y=self.avg_cosine_similarity_list
        
        plt.plot(X,Y)
        plt.xlabel("The number of clusters")
        plt.ylabel("The Avg cosine similarity score")
        plt.label("The number of culsters with the Avg cosine similarity score list")
        plt.show()
    
    
    def printOutTopicWords(self):
        result=TopicModeling(self.dataset,self.Best_topic_numbers,self.penality)
        result.Run_LDA_Model()   #result is the class
        final_model=result.LDA_model
        plot=result.visualize_LDA_model()
        return plot
    
    
    
           
    def text_process(self):
        for mess in self.sentences:
            partial_clean_word=[word.lower() for word in mess.split() if word not in stopwords.words("english")] #remove stop words
            remove_At_word=self.Words_Removed_At(partial_clean_word) #remove @people words 
            new_sentence=" ".join(remove_At_word)

            #remove the punctuation
            remove_punc=[char for char in new_sentence if char not in string.punctuation] #remove punctuation 
            remove_punc="".join(remove_punc)
            res_list=remove_punc.split(" ") 
            res_list=[word for word in res_list if word!="" and word.startswith("http")==False] #remove http 
            self.prepocessed_words.append(res_list)
    
    def Words_Removed_At(self,words): #remove @ people and sentences
        alist=[]
        for i in range(0,len(words)):
            word=words[i]
            if word.startswith("@")==False:
                alist.append(word)
        return alist
    
    
    def build_gram_model(self):
        bigram = gensim.models.Phrases(self.prepocessed_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[self.prepocessed_words], threshold=100)  

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        
        self.bigram_mod=bigram_mod
        self.trigram_mod=trigram_mod
        
    def make_bigrams(self):
        self.data_words_bigrams=[self.bigram_mod[doc] for doc in self.prepocessed_words]
    
    def make_trigrams(self):
        self.data_words_trigrams=[self.trigram_mod[self.bigram_mod[doc]] for doc in self.prepocessed_words]
        
    def lemmatization(self,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        nlp = spacy.load('en', disable=['parser', 'ner'])
        texts_out = []
        for sent in self.data_words_bigrams:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        self.data_lemmatized=texts_out
        
    
    
    def build_Id2Word_Corpus(self):
        self.id2word=corpora.Dictionary(self.data_lemmatized)
        self.corpus=[self.id2word.doc2bow(text) for text in self.data_lemmatized]
        
    def build_LDA_model(self):
        self.LDA_model=gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                           id2word=self.id2word,
                                           num_topics=self.number_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha="auto",
                                           per_word_topics=True)
    def getLDA_model(self):
        return self.LDA_model
    
        
        
    def visualize_LDA_model(self):
        pyLDAvis.enable_notebook()
        vis=pyLDAvis.gensim.prepare(self.LDA_model,self.corpus,self.id2word)
        self.plot=vis
        vis
        return vis
    
    
    
    
    def getAvgCosineSimilarity(self):
        Sum=0
        topics=self.LDA_model.print_topics()
        topics_number=len(topics)
        for j in range(0,len(topics)-1):
            sentence_1=self.getTopicScoreList(topics[j][1])
            for k in range(j+1,len(topics)):
                sentence_2=self.getTopicScoreList(topics[k][1])
                Sum+=self.getCosineSimilarity(sentence_1,sentence_2)
        avg_cosine=Sum/(topics_number*(topics_number-1)/2)
        avg_cosine+=(self.penality*topics_number)
        return avg_cosine
        
    
    def getPerplexity(self):
        perplexity=self.LDA_model.log_perplexity(self.corpus)
        return perplexity
    
    
    def getCoherence(self):
        coherence_model_lda = CoherenceModel(model=self.LDA_model,texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda
    
    
    
    #helper function to get Cosine similarity
    def getTopicScoreList(self,sentence):
        adict=dict()
        for atuple in sentence.split("+"):
            tuple_list=[]
            word=atuple.split("*")[1]
            number=float(atuple.split("*")[0])
            adict[word]=number
        return adict
    
    def getCosineSimilarity(self,sentence1,sentence2):
        Sum=0
        for key1 in sentence1:
            if key1 in sentence2:
                Sum+=sentence1[key1]*sentence2[key1]

        Sum_1=0
        for key1 in sentence1:
            Sum_1+=sentence1[key1]**2
        Sum_1=np.sqrt(Sum_1)

        Sum_2=0
        for key2 in sentence2:
            Sum_2+=sentence2[key2]**2
        Sum_2=np.sqrt(Sum_2)

        return Sum/(Sum_1*Sum_2)
    
    
    def getTopicFiveKeyWord(self):
        sentences=self.LDA_model.print_topics()
        all_topic=[]
        for i in range(0,len(sentences)):
            onesentence=sentences[i][1] #all string 
            number=0
            one_topic=""
            for j in range(0,len(onesentence.split("+"))):
                number_word=onesentence.split("+")[j]   #the format is like "0.082*"nepal" '
                mess_word=number_word.split("*")[1]
                clean_word=self.getCleanWord(mess_word)
                if number<5:
                    one_topic+=clean_word
                    one_topic+=" "
                    number+=1 
            all_topic.append(one_topic)
            
        #then we will print out every key word in the topic list
        
        self.topic_names=all_topic
    
    def getEachTopicNames(self):
        for m in range(0,len(self.topic_names)):
            print("The "+str(m+1)+" Topics Meanings are: "+str(self.topic_names[m]))
            
    
    def getCleanWord(self,aword):
        new_word=""
        for letter in aword:
            if letter!="'" and letter!='"' and letter!=" ":
                new_word+=letter
        return new_word 
        new_word=new_word.strip()
        
    
    def getEachSentenceLabel(self):
        label_list=[]

        for i in range(len(self.LDA_model[self.corpus])):
            atuple=self.LDA_model[self.corpus][i][0]

            max_value=0
            target_index=0

            for index, value in atuple:
                if value>max_value:
                    max_value=value
                    target_index=index
            label_list.append(self.topic_names[target_index])

        self.labels=label_list
    
    def getNewDataset(self):
        self.datasets["Labels"]=self.labels
        
        return self.datasets


# In[3]:


#inheritance then we can use the TopicModeling part method ,defualt topic is 1
class Mutiple_Times_LDA(TopicModeling):
    def __init__(self,dataset,alist,penality):
        self.dataset=dataset
        self.list=alist
        self.avg_cosine_similarity=[]
        self.perplexity_list=[]
        self.coherence_list=[]
        self.Best_topic_numbers=None
        self.penality=penality
        
    #optimize the algorithm because every time i am doing prepocessing over again
    #every time we only need to change the number of topics, Prepocessing part is same
    def runMutipleTimesLDA(self):
        lda_model=TopicModeling(self.dataset,self.penality)
        lda_model.Run_Prepocessing()
    
        
        for topic_number in self.list:
            lda_model.Run_LDA_Model(topic_number) #so it will call multiple times of every topics, we should not use the print
            lda_model_cosine=lda_model.getAvgCosineSimilarity()
            lda_model_perplexity=lda_model.getPerplexity()
            lda_model_coherence=lda_model.getCoherence()
            
            
            self.perplexity_list.append(lda_model_perplexity)
            self.coherence_list.append(lda_model_coherence)
            self.avg_cosine_similarity.append(lda_model_cosine)
            
        self.select_Best_Number_Topics()
        self.MakePlot_AvgCosine()
        self.MakePlot_Perplexity()
        self.MakePlot_Coherence()
        
        
        plot=self.printOutTopicWords()
        return plot
        
    
        
    
    def MakePlot_AvgCosine(self):
        X=self.list
        Y=self.avg_cosine_similarity
        
        plt.plot(X,Y)
        plt.xlabel("The number of clusters")
        plt.ylabel("The Avg cosine similarity score")
        plt.show()
        
        
        
    def MakePlot_Coherence(self):
        X=self.list
        Y=self.coherence_list
        
        plt.plot(X,Y)
        plt.xlabel("The number of clusters")
        plt.ylabel("The Coherence score")
        plt.show()
        
        
    def MakePlot_Perplexity(self):
        X=self.list
        Y=self.perplexity_list
        
        plt.plot(X,Y)
        plt.xlabel("The number of clusters")
        plt.ylabel("The Perplexity score")
        plt.show()
        
        
        
    def select_Best_Number_Topics(self):
        best=1 #the default value of the best number of topics
        best_number_topics=0
        for i in range(0,len(self.list)):
            number_topics=self.list[i]
            avg_cosine_similarity=self.avg_cosine_similarity[i]
            if avg_cosine_similarity<best:
                best=avg_cosine_similarity
                best_number_topics=number_topics
        self.Best_topic_numbers=best_number_topics
        
    
    def printOutTopicWords(self):
        result=TopicModeling(self.dataset,self.penality)
        result.Run_Prepocessing()
        result.Run_LDA_Model(self.Best_topic_numbers)   #result is the class
        result.getEachTopicNames()
        
        plot=result.visualize_LDA_model()
        return plot
    


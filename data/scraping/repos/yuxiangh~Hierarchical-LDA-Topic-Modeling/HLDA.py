
# coding: utf-8

# In[1]:


#implement the Hierarchical LDA topic modeling algorithmsc based on penalized avergae coisine similarity methods created by Yuxiang Hou
#problem desciption- LDA topic modeling algroithm always have a hard time to get the number of topics
#Yuxiang Hou implements the Auto-LDA algorithms to get the fewer and descriptive topics compared the traditional LDA topic modeling methods
#like perplexity, coherence.


#general idea- run the LDA topic modeling algorithm on the whole corpus, and then for a spefic corpus which are assigned topic meaning by Auto-LDA
#and then run LDA models again until tree depth reachs 5 or there is no topic related to disaster. (or Auto-LDA gives the Topic meaning "Others" to this corpus)
#make the Tree-Stuctured Hierarchical LDA modeling methods by doing like that.


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
#get_ipython().magic(u'matplotlib inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[2]:


import os 
import sys
import json
import pandas as pd
import numpy as np
import re
import time



#class of Hierarchical LDA is the main class that will get the input of json line path and lexicon path
# also the id path in a single, and the sentence path in the json files. 

class Hierarchical_LDA(object):
    def __init__(self,json_path,lexicon_path,id_path,sen_path):
        self.json_path=json_path
        self.lexicon_path=lexicon_path
        self.id_path=id_path
        self.sen_path=sen_path
        
        self.all_list=None
        self.dataframe=None
        self.lexicon=None #get the lexicon
        
    def runHierarchicalLDA(self):
        self.CreateDataFrame()
        self.getLexiconFileData()
        
        #create the LDA_Tree class
        lda_Tree=LDA_Tree(self.dataframe,self.lexicon)
        lda_Tree.create_Tree()
        lda_Tree.printOutHihierarchicalResult()
        
        
        res=lda_Tree.Layer_Meaning
        return res
    
    #the function in order to get the data from a json file, the sentence is "loreleiJSONMapping-translatedText", and it will give you the sentence. 
    def getWordsFromAlist(self,adict,aword):
        alist=aword.split(".")
        for i in range(0,len(alist)):
            new_dict=adict[alist[i]]
            adict=new_dict
        return adict
        
    
    #the function to get all json file in the json lines. 
    def getAllJsonFileData(self):
        all_list=[]
        with open(self.json_path,"r") as data_input:
            for line in data_input:
                one_list=[]
                data=json.loads(line)
                one_list.append(self.getWordsFromAlist(data,self.id_path)) 
                one_list.append(self.getWordsFromAlist(data,self.sen_path))
                all_list.append(one_list)
        self.all_list=all_list

    #create the dataframe
    def CreateDataFrame(self):
        result=self.getAllJsonFileData()
        column=["Id","Sentence"]
        data=pd.DataFrame(self.all_list,columns=column)
        self.dataframe=data
    
    #get all disaster realted terms from the lexicon including the topic and topic-related terms. 
    def getLexiconFileData(self):
        all_term=[]
        all_topic=set()
        with open(self.lexicon_path,"r") as data_input:
            data=json.load(data_input)
            for i in range(0,len(data)):
                sentence=data[i]
                all_term.append(sentence["term"].lower())
                all_topic.add(sentence["topic"].lower())
        disaster_words=set(all_term).union(set(all_topic))
        self.lexicon=disaster_words   
        
    
     
    
    


# In[4]:
#this module is the LDA topic modeling part
#including some prepocessing part(remove punctuation, stop words, words lemmitazation)
#also it includes  the method of average cosine similarity as a way to get the topic model numbers


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
    
    
    #LDA topic modeling part includeing getAvgCosineSimilarity method and getTopicTenKeyWord method
    def Run_LDA_Model(self,number_topics):
        self.number_topics=number_topics
        self.build_LDA_model()
        # self.visualize_LDA_model()
        self.getAvgCosineSimilarity()
        self.getTopicTenKeyWord()
        self.updateDataset()
        
        
    
    #select the number of topics based on average cosine similarity, the smaller of average cosine similarity, the better the results because it can 
    #seperate each topics very well.  
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
        
    
    #make the plot of average cosine similarity. 
    def MakePlot(self):
        X=self.topics_list
        Y=self.avg_cosine_similarity_list
        
        plt.plot(X,Y)
        plt.xlabel("The number of clusters")
        plt.ylabel("The Avg cosine similarity score")
        plt.label("The number of culsters with the Avg cosine similarity score list")
        plt.show()
    
    
    #print out the topic word 
    def printOutTopicWords(self):
        result=TopicModeling(self.dataset,self.Best_topic_numbers,self.penality)
        result.Run_LDA_Model()   #result is the class
        final_model=result.LDA_model
        plot=result.visualize_LDA_model()
        #return plot
    
    
    
    #text process including transfer to lower case, remove punctuation, replace http to url.     
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
    
    
    #build the gram model
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
    

    #words lemmatization. only allow the non, adj, verb, adv words. it will remove other types of words in a sentence. 
    def lemmatization(self,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        nlp = spacy.load('en', disable=['parser', 'ner'])
        texts_out = []
        for sent in self.data_words_bigrams:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        self.data_lemmatized=texts_out
        
    
    #build the id2word corpus
    def build_Id2Word_Corpus(self):
        self.id2word=corpora.Dictionary(self.data_lemmatized)
        self.corpus=[self.id2word.doc2bow(text) for text in self.data_lemmatized]


    #build the LDA topic modeling by using some empirical parameters.  
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
    
        
    #visualize the LDA model by the power of pyLDAvis package 
    def visualize_LDA_model(self):
        #pyLDAvis.enable_notebook()
        vis=pyLDAvis.gensim.prepare(self.LDA_model,self.corpus,self.id2word)
        self.plot=vis
        
        #return vis
    
    def getId2Word(self):
        return self.id2word
    
    def getCorpus(self):
        return self.corpus
    
    
    
    #get the average cosine similarity
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
        
    #get the perplexity of current model
    def getPerplexity(self):
        perplexity=self.LDA_model.log_perplexity(self.corpus)
        return perplexity
    
    #get the coherence score of the current model
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
    

    #get the cosine similarity of two sentences
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
    
    #get the top 10 key word
    def getTopicTenKeyWord(self):
        sentences=self.LDA_model.print_topics()
        all_topic=[]
        for i in range(0,len(sentences)):
            onesentence=sentences[i][1] #all string 
            number=0
            one_topic=[]
            for j in range(0,len(onesentence.split("+"))):
                number_word=onesentence.split("+")[j]   #the format is like "0.082*"nepal" '
                mess_word=number_word.split("*")[1]
                clean_word=self.getCleanWord(mess_word)
                if number<10:
                    one_topic.append(clean_word)
                    number+=1 
            one_topic=" ".join(one_topic)
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
    
        
        
    def getPercentageAndTopics(self):
        percent_Topic =self.LDA_model[self.corpus]
        percent_list=[]
        topic_list=[]

        for i in range(0,len(percent_Topic)):
            res=percent_Topic[i][0]

            highest_percent=0
            best_topic=None
            for j in range(0,len(res)):
                if res[j][1]>highest_percent:
                    highest_percent=res[j][1]
                    best_topic=self.topic_names[j]

            percent_list.append(highest_percent)
            topic_list.append(best_topic)
        return percent_list,topic_list
    
    
    def updateDataset(self):
        percent_list,topic_list=self.getPercentageAndTopics()
        self.datasets["Topic_Percent"]=percent_list
        self.datasets["Topic"]=topic_list


# In[5]:

#this module is mainly for solving the problem of how to get the correct number of LDA based on penalized average coisine similarity methods.
#main idea run LDA topic model from 2 to 20. The smaller the average cosine similarity, the better we can seperate each topics meaning. 

class Mutiple_Times_LDA(TopicModeling):
    def __init__(self,dataset,alist,penality):
        self.dataset=dataset
        self.list=alist
        self.avg_cosine_similarity=[]
        self.perplexity_list=[]
        self.coherence_list=[]
        self.Best_topic_numbers=None
        self.penality=penality
        self.final_model=None
        self.topic_names=[]
        
    
    #including lots of sub methods. 
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
        #self.MakePlot_AvgCosine()
        #self.MakePlot_Perplexity()
        #self.MakePlot_Coherence()
        self.getFinalModel()
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


    #get the final LDA topic model based on the lowest penalized average coisine similarity.    
    def getFinalModel(self):
        result=TopicModeling(self.dataset,self.penality)
        result.Run_Prepocessing()
        result.Run_LDA_Model(self.Best_topic_numbers)
        self.final_model=result
        self.topic_names=self.final_model.topic_names
        
        
    
    def printOutTopicWords(self):
        self.final_model.getEachTopicNames()
        plot=self.final_model.visualize_LDA_model()
        return plot
    
    #get the topic names 
    def getTopicNames(self):
        return self.topic_names
    
    
    def getListTopicName(self):
        all_list=[]
        for i in range(0,len(self.topic_names)):
            asentence=self.topic_names[i]
            alist=[]
            for j in range(0,len(asentence.split(" "))):
                word=asentence.split(" ")[j]
                alist.append(word)
            all_list.append(alist)
        return all_list
    
    
    def getNewDataSetDict(self):
        adict={}
        for topic in self.topic_names:
            new_data=self.dataset[self.dataset["Topic"]==topic]
            new_data=new_data.sort_values(by="Topic_Percent",ascending=False)
            new_data=new_data.drop(["Topic_Percent","Topic"],axis=1)
            adict[topic]=new_data
        return adict


# In[6]:
#this module is mainly for constructing a tree node
#a tree node will include the topic meaning, the number of twitter sentences
#the current depth, the parent's node, the top 10 most representative twitter sentences,and the dataset it included 

class TreeNode(object):
    def __init__(self,topic,number,cur_depth,parent,Top_10_Rep,dataset):
        self.topic=topic
        self.number=number
        self.cur_depth=cur_depth
        self.parent=parent
        self.Top_10_Rep=Top_10_Rep
        self.dataset=dataset


# In[7]:


class LDA_Tree(object):
    def __init__(self,dataset,lexicon):
        self.dataset=dataset #global variable
        self.max_depth=5
        self.lexicon=lexicon
        self.Layer_Meaning=dict()
        self.all_doc_number=dataset.shape[0]
        self.lexicon=lexicon
        
        
    def create_Tree(self):
        #two ways to terminate the recusive function
        #1.depth are over the max depth
        #2.The key word are not in Lexicon
        #3.if the current dataset len is fewer than 5% original dataset, stop splitting.
        
        #general idea---BFS algroithm using the queue to insert and pop the results. 
        
        queue=list()
        layer=1
        #initialize the queue and layer
        datasetTuple=("AllDoc",self.dataset)
        queue.append(datasetTuple)
        #construct the first Layer
        root=TreeNode("AllDoc",self.all_doc_number,0,None,None,self.dataset)
        self.Layer_Meaning[0]=[root]
        
        while len(queue)!=0 and layer<5:
            size=len(queue)
            self.Layer_Meaning[layer]=[]
            
            for i in range(0,size):
                dataset_tuple=queue.pop(0)
                dataset_meaning=dataset_tuple[0]
                dataset=dataset_tuple[1]
                
                multiple_LDA=Mutiple_Times_LDA(dataset,[x for x in range(2,11)],0.05)
                multiple_LDA.runMutipleTimesLDA()
                
                adict=multiple_LDA.getNewDataSetDict() #{XX:dataset1,XX:dataset2,XX:dataset3}
                
                for topic_name in adict:
                    son_dataset=adict[topic_name]
                    topic_name=self.transfer_Lexicon_Topic(topic_name)
                    if topic_name=="":
                        topic_name="Others"
                    
                    new_treeNode=TreeNode(topic_name,son_dataset.shape[0],layer,dataset_meaning,son_dataset.head(10),son_dataset)
                    if son_dataset.shape[0]!=0:
                        self.Layer_Meaning[layer].append(new_treeNode)
                    
                #then we have to junge whether this one will go to queue or not
                    if son_dataset.shape[0]>1 and topic_name!="Others":
                        son_dataset_tuple=(topic_name,son_dataset)
                        queue.append(son_dataset_tuple)
                        
            print("Then Enter The Next Layer")
            
            layer+=1
        
                    
    
    
    
    def transfer_Lexicon_Topic(self,asentence):
        alist=[]
        for word in asentence.split(" "):
            if word in self.lexicon:
                alist.append(word)
        return " ".join(alist)
    
    
    #print out each Layer's node result
    def printOutHihierarchicalResult(self):
        for layer_number in self.Layer_Meaning:
            node_list=self.Layer_Meaning[layer_number]

            for node in node_list:
                print("Current Depth is: "+str(node.cur_depth)+", Parent is :"+str(node.parent)+", Topic is: "+str(node.topic)+", Number is: "+str(node.number))
                print("===========================")
            print("Enter Next Layer")


# In[101]:

#main function 
if __name__=="__main__":
    start=time.time()
    
    #for the same layer we will merge the result
    #for the different,we will left join the layer
    #using the SQl similar left join/right join to get the final result

    path1=input("What's the path of json line? ")
    while path1=="":
        print("Please set up the correct json line path, the path cannot be empty")
        path1=input("What's the path of json line? ")
    

    path2=input("what's the path of disaster lexicon?")
    while path2=="":
        print("Please set up the correct lexicon path, the path cannot be empty")
        path2=input("what's the path of disaster lexicon?")


    Idpath=input("What's the path of Id in single json? ")
    while Idpath=="":
        print("Please set up the correct id path, the path cannot be empty")
        Idpath=input("what's the path of id path in a single json file?")

    
    SenPath=input("what's the path of sentence in single json? ")
    while SenPath=="":
        print("Please set up the correct sentence path in a single json file, the path cannot be empty")
        SenPath=input("what's the path of sentence path in a single json file?")


    

    def createTopRepresentative():
        os.makedirs("Hierarchical_LDA_Res")
        os.chdir("Hierarchical_LDA_Res")
        cur_dir=os.getcwd()
        
        for i in range(1,5):
            os.makedirs("Layer_"+str(i))
            os.chdir("Layer_"+str(i))                 
            cur_layer=res[i]
            
            for node in cur_layer:
                dataset_top_10=node.Top_10_Rep
                dataset_name=node.topic
                dataset_top_10.to_csv(str(dataset_name)+".csv")
            #then we will go back to previous layer
            os.chdir(cur_dir)
    
    def constructLayerDataset(cur_layer):
        alist=[]
        for node in cur_layer:
            dataset=node.dataset
            dataset_number=dataset.shape[0]
            dataset["Topic"]=[node.topic]*dataset_number
            alist.append(dataset)
        return pd.concat(alist)

    LDA_input=Hierarchical_LDA(path1,path2,Idpath,SenPath)
    res=LDA_input.runHierarchicalLDA()
    createTopRepresentative()
    
    
    layer1=constructLayerDataset(res[1])
    layer2=constructLayerDataset(res[2])
    layer3=constructLayerDataset(res[3])
    layer4=constructLayerDataset(res[4])

    layer1=layer1.rename(columns={"Topic":"First Topic"})
    layer2=layer2.rename(columns={"Topic":"Second Topic"})
    layer3=layer3.rename(columns={"Topic":"Third Topic"})
    layer4=layer4.rename(columns={"Topic":"Fourth Topic"})

    partial_1=pd.merge(layer1,layer2,how="left",on="Id")
    partial_1=partial_1.drop("Sentence_y",axis=1)
    partial_2=pd.merge(partial_1,layer3,how="left",on="Id")
    partial_2=partial_2.drop("Sentence",axis=1)

    final_dataset=pd.merge(partial_2,layer4,how="left",on="Id")
    final_dataset=final_dataset[["Id","Sentence_x","First Topic","Second Topic","Third Topic","Fourth Topic"]]
    final_dataset=final_dataset.rename(columns={"Sentence_x":"Sentence"})
    final_dataset.to_csv("Hierarchical_LDA_Dataset.csv")
    
    
    print("Mission Success! Hierarchical LDA algroithms completed!")
    end=time.time()

    print("The total Running time of Hierarchical LDA is: "+str(end-start)+" Seconds")
    

# for each layer, we will merge the result.
# and for each different layer, we will left join

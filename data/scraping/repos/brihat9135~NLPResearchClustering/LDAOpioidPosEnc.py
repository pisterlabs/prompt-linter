# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:42:21 2018

@author: cs
"""
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.cluster import KMeans
import gensim
import random as rn
rn.seed(29)
np.random.seed(1337)
import re
from UMLSconnect import UMLSClient
from gensim.models.coherencemodel import CoherenceModel


"""
Not all the function from the class are used
"""

class ReadTxtFiles:
    
    def listDirectoryFiles(a):
        txtfilesTrain = []
        count = 0
        for file in os.listdir(a):
            try:
                if file.endswith(".txt"):
                    txtfilesTrain.append(a + "/" + str(file))
                    count = count + 1
                else:
                    print("There is no text file")
            except Exception as e:
                raise e
                print("No files found here!")
        print("Total files found:", count)
        return txtfilesTrain
    
    def listDirectoryFiles1(path):
        txtfilesTrain = []
        count = 0
        for path, subdirs, files in os.walk(path):
            #print(path)
            for name in files:
                #print(os.path.join(path, name))
                new_path = os.path.join(path, name)
                txtfilesTrain.append(new_path)
        return txtfilesTrain
    
    def convertToDF(a):
        df_train = pd.DataFrame(a, columns = ['filepath'])
        return df_train
        
        
    def transformData(data):
        """
        This function can be used if you want to bag of words or CUIS
        """
        #count_vect = CountVectorizer()
        
        #X_train_counts = count_vect.fit_transform(data)
        tf_vectorizer = TfidfVectorizer(norm = 'l2', use_idf=False, min_df = 200, max_df = 10000, ngram_range = (1, 2))
        X_train_vec = tf_vectorizer.fit_transform(data)
        #print(X_train_tfdif)
        return X_train_vec


    def giveDir(x):
        loc = "/Archive/Opioid/OpioidLDA/pos_CUIS/"
        loc = loc + str(x) + ".txt"
        if os.path.isfile(loc):
            return loc
        return "No"
                        

    
    
if __name__ == '__main__':
    
    RTF = ReadTxtFiles
    
    """
    -these are all read data part so modify accordingly, the code here is reading list of encounter from a file
    -provide directory where the notes for each encounter lies and open them all at once
    -you can use listdirectoryFiles function from the above class to get files from the folders directory if that's easier
    """

    path = "/home/bsharma1/Opioid/LDA_Opioid/lca_5class_assignments.csv"
    data_df = pd.read_csv(path, sep = ',')
    
    #providing directory to each encounter so we can open them at once
    data_df['hsp_account_id'] = data_df.hsp_account_id.apply(lambda x: RTF.giveDir(x))

    print(data_df.head(10))
    data_df = data_df[data_df.hsp_account_id != "No"]
    #data_df = data_df[data_df['class'] == 4]
    
    
    data_df['hsp_account_id'] = data_df.hsp_account_id.apply(lambda x: open(x, "r").read())
    #data_df['filepath'] = data_df.filepath.apply(lambda x: ", ".join(x.split( )))
    data_df['hsp_account_id'] = data_df.hsp_account_id.apply(lambda x: x.split( ))


    """
    #this is where you will start processing the data and apply LDA technique
    """
    
    documents = data_df['hsp_account_id'].tolist() 
    dictionary = gensim.corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below = 190, no_above = 0.7)
    dictionary.save("saveCorporaDicV4_20_250")
    corpus = [dictionary.doc2bow(text) for text in documents]
    
    #this is where you will apply the actual model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 20, id2word = dictionary, passes = 250)

    ldamodel.save('OpioidLdamodelV4_20_250.lda')
    lda = gensim.models.ldamodel.LdaModel.load('SUD_20_250_70.lda')

    cm = CoherenceModel(model = lda, texts = documents, corpus = corpus, coherence = 'c_v')
    coherence = cm.get_coherence()

    ldaP = lda.print_topics(num_topics = 20, num_words = 25)
   
    print(coherence)    
    


    
    




    
    
    
    

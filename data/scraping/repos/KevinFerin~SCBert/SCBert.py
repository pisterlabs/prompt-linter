#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:49:32 2020

Sentence Clustering with pretrained BERT models

@author: kevin
"""
import torch
from transformers import FlaubertModel,FlaubertTokenizer, FlaubertConfig
from transformers import CamembertModel,CamembertTokenizer, CamembertConfig
import numpy as np 
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from stop_words import get_stop_words
from multi_rake import Rake
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from string import punctuation

MODELS = {"flaubert_base" : {
                  "model" : FlaubertModel,
                  "tokenizer" : FlaubertTokenizer,
                  "config" : FlaubertConfig, 
                  "nb_layer" : 12,
                  "hidden_length" : 768,
                  "pad_id" : 2,
                  "model_name" : 'flaubert-base-uncased'},
          "flaubert_large" : {
                  "model" : FlaubertModel,
                  "tokenizer" : FlaubertTokenizer,
                  "config" : FlaubertConfig, 
                  "nb_layer" : 24,
                  "hidden_length" : 1024,
                  "pad_id" : 2,
                  "model_name" : 'flaubert-large-cased'},
          "flaubert_small" : {
                  "model" : FlaubertModel,
                  "tokenizer" : FlaubertTokenizer,
                  "config" : FlaubertConfig, 
                  "nb_layer" : 6,
                  "hidden_length" : 512,
                  "pad_id" : 2,
                  "model_name" : 'flaubert-small-cased'},
          "camembert" : {
                  "model" : CamembertModel,
                  "tokenizer" : CamembertTokenizer,
                  "config" : CamembertConfig, 
                  "nb_layer" : 12,
                  "hidden_length" : 768,
                  "pad_id" : 1,
                  "model_name" : 'camembert-base'}}

class Vectorizer :
        
        def __init__(self, model_name = "flaubert_base") :
            """
            Constructor of the vectorizer object used to transform your texts into vectors using french BERT models. 

            Parameters
            ----------
            model_name : string, optional
                Corresponds to the model you want to use to tokenize and vectorize your data. Only CamemBERT and Flaubert small, base and large for now. 
                DESCRIPTION. The default is "flaubert_base".
                
            ----------

            """
            self.model_dict = MODELS[model_name.lower()]
            self.model = self.model_dict["model"].from_pretrained(self.model_dict["model_name"],  output_hidden_states=True)
            self.tokenizer = self.model_dict["tokenizer"].from_pretrained(self.model_dict["model_name"])
            self.pad_id = self.model_dict["pad_id"]
            self.nb_layer = self.model_dict["nb_layer"]
            self.hidden_length = self.model_dict["hidden_length"]
        
        
        def tokenize (self, data, MAX_LEN = 256) :
            """
            This function call the tokenizer corresponding to the BERT model specified in the constructor. Then it generates
            a vector of id corresponding to the words in the vocabulary. Also an attention vector which has the same size as the vector of id with ones 
            for real words and zeros corresponding to padding id. 
            
            Parameters
            ----------
            data : `Numpy array` or `Pandas DataFrame`
                Corresponds to your datas, must be a list of your texts texts. 
                
            MAX_LEN : int, optional
                Corresponds to the max number of word to take into account during tokenizing. If a text is 350 words long and 
                MAX_LEN is 256, the text will be truncated after the 256th word, starting at the beginning of the sentence. 
                Default: The default is 256.
                

            Returns
            -------
            tokenized_texts : List of list of strings
                Corresponds of the list of your tokenized texts. Each text has been transformed into a vector of word according the tokenizer of the BERT model stated into the constructor.
            input_ids_tensor : List of List of int
                Same as tokenized_texts but with ids corresponding to the tokens, converted into torch tensor. 
            masks_tensor : List of list of float
                Corresponds to the attention torch tensor
            """
            tokenized_texts = np.array([self.tokenizer.tokenize(text) for text in data])
            input_ids = np.array([self.tokenizer.encode(text, max_length=MAX_LEN, pad_to_max_length=True,  add_special_tokens= True ) for text in data])
            # Create attention masks
            attention_masks = []
          
            # Create a mask of 1s for each token followed by 0s for padding
            for seq in input_ids:
                seq_mask = [float(i != self.pad_id) for i in seq]
                attention_masks.append(seq_mask)
          
            # Convert all of our data into torch tensors, the required datatype for our model
            input_ids_tensor = torch.tensor(input_ids)
            masks_tensor = torch.tensor(attention_masks)
            
            return tokenized_texts, input_ids_tensor, masks_tensor
        
        def __sentence_pooling (self, vectors , pooling_method) :
            """
            Parameters
            ----------
            vectors : list of vectors representing each words including the BOS and EOS tag
            
            pooling_method : string
                average or max.

            Returns
            -------
            pooled_vectors : tensor 
                pooled tensors according to the method.

            """

            pooled_vector = torch.tensor([])
            if pooling_method.lower() == "average" :
                pooled_vector = torch.mean(vectors, axis=0)
                
            elif pooling_method.lower() == "max" :
                pooled_vector = torch.max(vectors, axis=0)
            
            return pooled_vector
        
        def __word_pooling (self, encoded_layers_b, layers, idx,  pooling_method, MAX_LEN = 256) :
            """
            Parameters
            ----------
            vectors : list of vectors representing each words including the BOS and EOS tag
            
            pooling_method : string
                average, max or concat.
                
            MAX_LEN : int, optional
                Corresponds to the max number of word to take into account during tokenizing. If a text is 350 words long and 
                MAX_LEN is 256, the text will be truncated after the 256th word, starting at the beginning of the sentence. 
                Default: The default is 256.

            Returns
            -------
            pooled_words : tensor 
                pooled tensors according to the method.

            """
            pooled_words = torch.tensor([])
            
            if pooling_method.lower() == "concat" :
                for layer in layers :   
                    pooled_words = torch.cat((pooled_words, encoded_layers_b[layer][idx]), dim=1)
                    
            if pooling_method.lower() == "average" :
                pooled_words = torch.tensor([[0. for i in range(self.hidden_length)] for j in range(MAX_LEN)])
                for layer in layers :
                    pooled_words = pooled_words.add(encoded_layers_b[layer][idx])
                pooled_words = pooled_words/(len(layers))
                
            elif pooling_method.lower() == "max" :
                pooled_words = torch.tensor([[-100. for i in range(self.hidden_length)] for j in range(MAX_LEN)])
                for layer in layers :   
                    pooled_words = torch.max(pooled_words, encoded_layers_b[layer][idx])
            
            return pooled_words
        
        def __batch(self, iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]
                
        def __test():
            return 
                
        def forward_and_pool (self, input_ids_tensor, masks_tensor, sentence_pooling_method="average", word_pooling_method="average", layers = 11, batch_size=50, path_to_save=None) :
            """
            This function execute the forward pass of the input data into the BERT model and create a unique tensor for each input according to the stated pooling methods. 
            
            Parameters
            ----------
            input_ids_tensor : tensor
                Corresponds to then= ids of tokenized words. 
                Must match the output of the tokenize function.
            masks_tensor : tensor
                Corresponds to the attention masks of tokenized words. 
                Must match the output of the tokenize function. .
            sentence_pooling_method : str, optional
                Corresponds to the method of pooling to create a unique vector for each text
                The default is "average".
            word_pooling_method : str, optional
                Corresponds to the method of word pooling method in the case multiple layers have been stated. In other words it is the way to compute a single word vector from multiple layers.
                The default is "average".
            layers : int or list, optional
                Corresponds to the BERT layers to use to create the embeddings of words.
                The default is 11.
            batch_size : int, optional
                Size of batch when executing the forward pass into the BERT model, should get lower as your computational power gets lower. 
                The default is 50.
            path_to_save : str or None, optional
                Is the path to save the vector of texts if you want to avoid doing this computation again as it may be long. 
                if None, nothing will be saved. 
                The default is None.
            
            Returns
            -------
            texts_vectors : list
                A list of tensors, each tensor corresponding to an input text.

            """
            
            layer_list = False
            if (sentence_pooling_method not in ["average", "max"]) :
                raise ValueError('sentence_pooling_method must be equal to `average` or `max`')
                
            if (word_pooling_method not in ["average", "max", "concat"]) :
                raise ValueError('word_pooling_method must be equal to `average`, `max` or `concat` ')
            
            if(type(batch_size) != int) :
                raise TypeError('batch_size must be a positive integer')
                
            if(batch_size<=0) :
                raise ValueError('batch_size must be a positive integer')
                
            if((type(path_to_save)  != str ) and (path_to_save != None)):
                raise TypeError('path_to_save must be None or a string')
                
            if (type(layers) != int) :
                if (type(layers) == list) :
                    layer_list = True
                    for el in layers : 
                        if (type(el) != int) :
                            raise TypeError('layers must be a int between 1 and nb_layer or a list of integers between 1 and nb_layer')
                        elif (el>self.nb_layer or el<1) :
                            raise ValueError('layers must be a int between 1 and nb_layer or a list of integers between 1 and nb_layer')
                else :
                    raise TypeError('layers must be a int between 1 and nb_layer or a list of integers between 1 and nb_layer')
            else :
                if (layers>self.nb_layer or layers<1) :
                    raise ValueError('layers must be a int between 1 and nb_layer or a list of integers between 1 and nb_layer')
            
            texts_vectors = []
            N = len(input_ids_tensor)
            counter = 0
            with tqdm(total = 100) as pbar : 
                for b in self.__batch(range(0,N), batch_size) :
                    with torch.no_grad() :
                        encoded_layers_b = self.model(input_ids_tensor[b], masks_tensor[b].to(torch.int64))[1]
                            
                        if layer_list :
                            for idx in b :
                                if input_ids_tensor[idx][-1]==1 :
                                    eos_pos = 0
                                else :
                                    eos_pos = int((input_ids_tensor[idx] == self.pad_id).nonzero()[0])
                                word_vector = self.__word_pooling(encoded_layers_b, layers, idx - counter, word_pooling_method)
                                pooled_vector = self.__sentence_pooling(word_vector[:eos_pos-1][1:], sentence_pooling_method) #Just no to take into account BOS and EOS 
                                texts_vectors.append(pooled_vector)
                            
                        else : 
                            words_vector = encoded_layers_b[layers]
                            
                            for idx in b : 
                                if input_ids_tensor[idx][-1]==1 :
                                    eos_pos = 0
                                else :
                                    eos_pos = int((input_ids_tensor[idx] == self.pad_id).nonzero()[0])
                                pooled_vector = self.__sentence_pooling(words_vector[idx-counter][:eos_pos-1][1:], sentence_pooling_method) #Just no to take into account BOS and EOS 
                                texts_vectors.append(pooled_vector)
                    counter += batch_size
                    pbar.update(np.round(100*len(b)/N,2))
            
            if path_to_save != None : 
              torch.save(texts_vectors, path_to_save+"text_vectors.pt")
            
            return texts_vectors
        


        def vectorize (self, data, MAX_LEN = 256,sentence_pooling_method="average", word_pooling_method="average", layers = 11, batch_size=50, path_to_save=None) :
            """
            Transform the input raw data into tensors according to the selected models and the pooling methods. 
            
            Parameters
            ----------
            data : `Numpy array` or `Pandas DataFrame`
                Corresponds to your datas, must be a list of your texts texts. 
                
            MAX_LEN : int, optional
                Corresponds to the max number of word to take into account during tokenizing. If a text is 350 words long and 
                MAX_LEN is 256, the text will be truncated after the 256th word, starting at the beginning of the sentence. 
                Default: The default is 256.
            sentence_pooling_method : str, optional
                Corresponds to the method of pooling to create a unique vector for each text
                The default is "average".
            word_pooling_method : str, optional
                Corresponds to the method of word pooling method in the case multiple layers have been stated. In other words it is the way to compute a single word vector from multiple layers.
                The default is "average".
            layers : int or list, optional
                Corresponds to the BERT layers to use to create the embeddings of words.
                The default is 11.
            batch_size : int, optional
                Size of batch when executing the forward pass into the BERT model, should get lower as your computational power gets lower. 
                The default is 50.
            path_to_save : str or None, optional
                Is the path to save the vector of texts if you want to avoid doing this computation again as it may be long. 
                if None, nothing will be saved. 
                The default is None.

            Returns
            -------
            texts_vectors : list
                A list of tensors, each tensor corresponding to an input text.

            """
            tokenized_texts, input_ids_tensor, masks_tensor = self.tokenize(data,MAX_LEN)
            texts_vectors = self.forward_and_pool(input_ids_tensor,masks_tensor,sentence_pooling_method,word_pooling_method,layers,batch_size,path_to_save)

            return texts_vectors
        
class EmbeddingExplorer :
    
    def __init__(self,data, texts_vectors) :
        self.data = data
        if type(texts_vectors) == str : 
            self.texts_vectors = np.array([el.tolist() for el in torch.load(texts_vectors)])
        else : 
            self.texts_vectors = np.array([el.tolist() for el in texts_vectors])
        self.labels = [0 for i in range (len(texts_vectors))]
        self.keywords = {}

    def cluster (self, k, cluster_algo="k-means") :
        if cluster_algo=="k-means" :
            clf = KMeans(n_clusters=k,
                  max_iter=50,
                  init='k-means++',
                  n_init=4)
            self.labels = clf.fit_predict(self.texts_vectors)
            
        elif cluster_algo=="quick_k-means" :
            clf = MiniBatchKMeans(n_clusters=k,
                  max_iter=50,
                  init='k-means++',
                  n_init=4)
            self.labels = clf.fit_predict(self.texts_vectors)
        
        elif cluster_algo=="dbscan" :
            clf = DBSCAN()
            self.labels = clf.fit_predict(self.texts_vectors)
            
        elif cluster_algo=="agglomerative" :
            clf = AgglomerativeClustering(n_clusters=k)
            self.labels = clf.fit_predict(self.texts_vectors)
            
        elif cluster_algo=="spectral":
            clf = SpectralClustering(n_clusters=k, assign_labels="discretize")
            self.labels = clf.fit_predict(self.texts_vectors)
            
        return self.labels
    
    
    def extract_keywords(self, max_words = 1, min_freq=5, num_top_words=10) :

        stop_words = get_stop_words('fr')
        
        rake = Rake(max_words=max_words, min_freq = min_freq, language_code ="fr", stopwords = stop_words)
        
        for i, label in enumerate(np.unique(self.labels)):
              corpus_fr = ' '.join(self.data[self.labels==label])
              keywords = rake.apply(corpus_fr)
              top_words= np.array(keywords[:num_top_words])[:,0]
              self.keywords["Cluster {0}".format(label)] = top_words
              
        return self.keywords
    
    def compute_coherence(self, vectorizer, keywords=[]) :
        if not isinstance(vectorizer, Vectorizer) :
            raise TypeError("You should provide the same Vectorizer object you used to compute the vectors")
        tokenized_texts = np.array([vectorizer.tokenizer.tokenize(text) for text in self.data])
        coherences =[]
        if keywords == [] :
            if self.keywords == {} :
                raise ValueError('There is no keywords extracted or passed in the function')
            else : 
                for cluster, top_words in enumerate(list(self.keywords.values())) :
                    cm = CoherenceModel(topics=top_words, texts=tokenized_texts, dictionary=Dictionary(tokenized_texts) , coherence="c_v")
                    coherences.append(cm.get_coherence())
                    print("Cluster {} with keywords : \n {} \n has a coherence of {} \n".format(cluster, top_words, cm.get_coherence()))
                return coherences
        else : 
            for cluster, top_words in enumerate(keywords) :
                cm = CoherenceModel(topics=top_words, texts=tokenized_texts, dictionary=Dictionary(tokenized_texts) , coherence="c_v")
                coherences.append(cm.get_coherence())
                print("Cluster {} with keywords : \n {} \n has a coherence of {} \n".format(cluster, top_words, cm.get_coherence()))
            return coherences
            
    def extract_keywords_and_coherence(self, vectorizer, keywords = [], max_words =1, min_freq=5, num_top_words=10) :
        if not isinstance(vectorizer, Vectorizer) :
            raise TypeError("You should provide the same Vectorizer object you used to compute the vectors")
        keywords = self.extract_keywords(max_words =1, min_freq=5, num_top_words=10)
        
        return keywords, self.compute_coherence(vectorizer, keywords)
    
    # def __custom_cls_metric(self) :
        
    #     s = 0
    #     p1,p2,p3=[],[],[]
    #     for i in range (3):
    #         s+=round(sum(self.labels==i)/len(self.labels),3)
    #         p1.append(sum(self.labels[:2000] == i)/2000)
    #         p2.append(sum(self.labels[2000:4000] == i)/2000)
    #         p3.append(sum(self.labels[4000:] == i)/2000)
    #     p = [p1,p2,p3]
    #     index_to_chose = [0,1,2]
    #     for i in range (3):
    #         l=[]
    #         for j in range (3) :
    #             if (j in index_to_chose) :
    #                 l.append(p[i][j])
    #             else : l.append(0)
    #         #l = [p[i][j] for j in range (3) if j in index_to_chose]
    #         s+=max(l)
    #         index_to_chose.remove(l.index(max(l)))
    #     return 4-s
    
    def explore_cls(self, color_label, projection_method= "PCA") :
        if projection_method=="PCA" :
            
            pca = PCA(n_components=2).fit(self.texts_vectors)
            datapoint = pca.transform(self.texts_vectors)
            
            plt.figure(figsize=(10, 10))
            plt.title("PCA representation of the cls data after vectoring with BERT", fontsize=15)
            plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color_label, cmap='Set1' )
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.show()
            
        elif projection_method=="tSNE" :
            datapoint = TSNE(n_components=2).fit_transform(self.texts_vectors)
            plt.figure(figsize=(10, 10))
            plt.title("tSNE representation of the cls data after vectoring with BERT", fontsize=15)
            plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color_label, cmap='Set1' )
            plt.xlabel("dim 1")
            plt.ylabel("dim 2")
            plt.show()
            
        p_1,p_2,p_3=[], [], []
        for i in range(3) :
            print("Proportion of cluster {} : {}".format(i,round(sum(self.labels==i)/len(self.labels),3)))
            p_1.append(sum(self.labels[:2000] == i))
            p_2.append(sum(self.labels[2000:4000] == i))
            p_3.append(sum(self.labels[4000:] == i))
        p = [p_1,p_2,p_3]
        plt.figure(figsize=(15,15))
        category = ["DVD", "Musique", "Livres"]
        for i in range (1,4) :
            plt.subplot(1,3,i)
            plt.title("Proportion of each cluster within {}".format(category[i-1]))
            plt.pie(p[i-1], labels=[0,1,2])
        plt.show()
        
        #print("Our custom metric gives us {}. \nCloser to 0 is better".format(self.__custom_cls_metric()))

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:10:28 2021
TopicModel+ class definition
@author: srandrad, hswalsh
"""

import pandas as pd
import tomotopy as tp
import numpy as np
from time import time,sleep
from tqdm import tqdm
import os
import datetime
import pyLDAvis
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

class Topic_Model_plus():
    """ Topic model plus

    A class for topic modeling for aviation safety.
    
    Attributes
    ----------
    text_columns: list
        defines various columns within a single database which will be used for topic modeling
    data : Data
        Data object storing the text corpus
    ngrams : str
        'tp' if the user wants tomotopy to form ngrams prior to applying a topic model 
    doc_ids : list
        list of document ids pulled from data object
    data_df : pandas dataframe
        df storing documents pulled from data object
    data_name : string
        dataset name pulled from data object
    id_col : string
        the column storing the document ids pulled from data object
    hlda_models : dictionary
        variable for storing hlda models
    lda_models : dictionary
        variable for storing lda models
    folder_path : string
        destination for storing results and models
    results_path : string
        destentation to create results folder in
        
    """

#   TO DO:
#   add hyper parameter tuning for lda (alpha and beta and k/number of topics) and hlda (eta, alpha, gamma)
    
    def __init__(self, text_columns=[], data=None, ngrams=None, results_path=''):
        """ 

        CLASS CONSTRUCTORS
        ------------------
        text_columns: list
            defines various columns within a single database which will be used for topic modeling
        data : Data
            Data object storing the text corpus
        ngrams : str
            'tp' if the user wants tomotopy to form ngrams prior to applying a topic model 
        results_path : str
            location to create results folder.
        """
        self.text_columns = text_columns
        self.data = data
        self.doc_ids = data.doc_ids
        self.data_df = data.data_df
        self.data_name = data.name
        self.id_col = data.id_col
        self.hlda_models = {}
        self.lda_models = {}
        self.BERT_models = {}
        self.reduced = False
        self.ngrams = ngrams
        self.folder_path = ""
        if results_path != '':
            self.results_path = results_path + '/'
        else:
            self.results_path = results_path
        
    def __create_folder(self): 
        """
        makes a folder/directory for saving results

        Returns
        -------
        None.

        """
        #new version, just saves it within the folder
        if self.folder_path == "":
            self.folder_path = self.results_path+ 'topic_model_results'
            if os.path.isdir(self.folder_path) == True:
                today_str = datetime.date.today().strftime("%b-%d-%Y")
                self.folder_path += today_str
            os.makedirs(self.folder_path, exist_ok = True)
    
    def get_bert_coherence(self, coh_method='u_mass', from_probs=False):
        """
        Gets coherence for bert models and saves it in a dictionary.

        Parameters
        ----------
        coh_method : string, optional
            Method used to calculate coherence. Can be any method used in gensim. 
            The default is 'u_mass'.
        from_probs : boolean, optional
            Whether or not to use document topic probabilities to assign topics.
            True to use probabilities - i.e., each document can have multiple topics.
            False to not use probabilities - i.e., each document only has one topics.
            The default is False.

        Returns
        -------
        None.

        """

        self.BERT_coherence = {}
        for col in self.text_columns:
            if from_probs == False: #each document only has one topic
                docs = self.data_df[col].tolist()
                topics = self.BERT_model_topics_per_doc[col]
            elif from_probs == True: #documents can have multiple topics
                docs = []; topics = []
                text = self.data_df[col].tolist()
                for doc in range(len(self.BERT_model_all_topics_per_doc[col])):
                    for k in self.BERT_model_all_topics_per_doc[col][doc]:
                        topics.append(k)
                        docs.append(text[doc])
            topic_model = self.BERT_models[col]
            self.BERT_coherence[col] = self.calc_bert_coherence(docs, topics, topic_model, method=coh_method)

    def calc_bert_coherence(self, docs, topics, topic_model, method='u_mass', num_words=10):
        """
        Calculates coherence for a bertopic model using gensim coherence models.

        Parameters
        ----------
        docs : list
            List of document text.
        topics : List
            List of topics per document.
        topic_model : BERTopic model object
            Object containing the trained topic model.
        method : string, optional
            Method used to calculate coherence. Can be any method used in gensim. 
            The default is 'u_mass'.
        num_words : int, optional
            Number of words in the topic used to calculate coherence. The default is 10.

        Returns
        -------
        coherence_per_topic : List
            List of coherence scores for each topic.

        """

        # Preprocess Documents
        documents = pd.DataFrame({"Document": docs,
                                  "ID": range(len(docs)),
                                  "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
        
        # Extract vectorizer and analyzer from BERTopic
        vectorizer = topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names_out()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        ordered_topics = list(set(topics))
        ordered_topics.sort()
        topic_words = [[words for words, _ in topic_model.get_topic(topic)[:num_words]]
                       for topic in ordered_topics]
        # Evaluate
        coherence_model = CoherenceModel(topics=topic_words,
                                         texts=tokens,
                                         corpus=corpus,
                                         dictionary=dictionary,
                                         coherence=method)
        coherence_per_topic = coherence_model.get_coherence_per_topic()
        return coherence_per_topic
    
    def save_bert_model(self, embedding_model=True):
        """
        Saves a BERTopic model.

        Parameters
        ----------
        embedding_model : boolean, optional
            True to save the embedding model. The default is True.

        Returns
        -------
        None.

        """

        self.__create_folder()
        for col in self.text_columns:
            path = "_BERT_model_object.bin"
            if self.reduced: path = "_Reduced" + path
            self.BERT_models[col].save(os.path.join(self.folder_path,col+path),save_embedding_model=embedding_model)
        self.data.save(results_path=os.path.join(self.folder_path,"preprocessed_data.csv"))
    
    def load_bert_model(self, file_path, reduced=False, from_probs=False, thresh=0.01):
        """
        Loads trained bertopic model(s).

        Parameters
        ----------
        file_path : string
            file path to the folder storing the model(s)
        reduced : bool, optional
            True if the model is a reduced topic model, by default False
        from_probs : bool, optional
            Whether or not to use document topic probabilities to assign topics.
            True to use probabilities - i.e., each document can have multiple topics.
            False to not use probabilities - i.e., each document only has one topics, by default False
        thresh : float, optional
            probability threshold used when from_probs=True, by default 0.01
        """

        self.BERT_models = {}
        self.BERT_model_topics_per_doc = {}
        self.BERT_model_probs={}
        self.BERT_model_all_topics_per_doc={}
        for col in self.text_columns:
            path = "_BERT_model_object.bin"
            if reduced: 
                path = "_Reduced" + path
                self.reduced = True
            self.BERT_models[col] = BERTopic.load(os.path.join(file_path,col+path))
            self.BERT_model_topics_per_doc[col] = self.BERT_models[col].topics_
            self.BERT_model_probs[col] = self.BERT_models[col].probabilities_
            if from_probs == True:
                self.__get_all_topics_for_doc(col, self.BERT_model_probs[col], thresh, self.BERT_model_topics_per_doc[col])
        preprocessed_filepath = os.path.join(file_path,"preprocessed_data")
        self.data.load(preprocessed_filepath+".csv", preprocessed=True, id_col=self.data.id_col, text_columns=self.data.text_columns)
        self.folder_path = file_path

    def save_bert_coherence(self, return_df=False, coh_method='u_mass', from_probs=False):
        """
        Saves the coherence scores for a bertopic model.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the coherence_df. The default is False.
        coh_method : string, optional
            Method used to calculate coherence. Can be any method used in gensim. 
            The default is 'u_mass'.
        from_probs : boolean, optional
            Whether or not to use document topic probabilities to assign topics.
            True to use probabilities - i.e., each document can have multiple topics.
            False to not use probabilities - i.e., each document only has one topics.
            The default is False.

        Returns
        -------
        coherence_df : pandas DataFrame
            Dataframe with each row a topic, column has coherence scores.

        """

        self.get_bert_coherence(coh_method, from_probs)
        self.__create_folder()
        max_topics = max([len(set(self.BERT_models[col].topics_))-1 for col in self.BERT_models])
        coherence_score = {"topic numbers": ["average score"]+['std dev']+[i for i in range(-1, max_topics)]}
        for col in self.text_columns:
            coherence_score[col] = []
            c_scores = self.BERT_coherence[col]
            average_coherence = np.average(c_scores)
            coherence_score[col].append(average_coherence)
            std_coherence = np.std(c_scores)
            coherence_score[col].append(std_coherence)
            coherence_per_topic = c_scores
            for i in range(-1, (max_topics-len(coherence_per_topic))):
                coherence_per_topic.append("n/a")
            coherence_score[col] += coherence_per_topic
        coherence_df = pd.DataFrame(coherence_score)
        if return_df == True:
            return coherence_df
        path = "BERT_coherence_"
        if self.reduced: path = "Reduced_" + path
        coherence_df.to_csv(os.path.join(self.folder_path,path+coh_method+".csv"))
    
    def get_bert_topic_diversity(self, topk=10):
        """
        Gets topic diversity scores for a BERTopic model.

        Parameters
        ----------
        topk : int, optional
            Number of words per topic used to calculate diversity. The default is 10.

        Returns
        -------
        None.

        """

        topic_diversity = TopicDiversity(topk=topk)
        self.diversity = {col: [] for col in self.text_columns}
        for col in self.text_columns:
            ordered_topics = list(set(self.BERT_models[col].topics_))
            ordered_topics.sort()
            output = {'topics': [[words for words,_ in self.BERT_models[col].get_topic(topic)] 
                                 for topic in ordered_topics]}
            score = topic_diversity.score(output)
            self.diversity[col].append(score)
    
    def save_bert_topic_diversity(self, topk=10, return_df=False):
        """
        Saves topic diversity score for a bertopic model.
        
        Parameters
        ----------
        topk : int, optional
            Number of words per topic used to calculate diversity. The default is 10.
        return_df : boolean, optional
            True to return the diversity_df. The default is False.

        Returns
        -------
        diversity_df : pandas DataFrame
            Dataframe with topic diversity score.

        """

        self.get_bert_topic_diversity(topk=10)
        diversity_df = pd.DataFrame(self.diversity)
        if return_df == True:
            return diversity_df
        path = "BERT_diversity"
        if self.reduced: path = "Reduced_" + path
        diversity_df.to_csv(os.path.join(self.folder_path,path+".csv"))
        
    def bert_topic(self, sentence_transformer_model=None, umap=None, hdbscan=None, count_vectorizor=None, ngram_range=(1,3), BERTkwargs={}, from_probs=True, thresh=0.01):
        """
        Train a bertopic model.

        Parameters
        ----------
        sentence_transformer_model : BERT model object, optional
            BERT model object used for embeddings. The default is None.
        umap : umap model object, optional
            umap model object used for dimensionality reduction. The default is None.
        hdbscan : hdbscan model object, optional
            hdbscan model object for clustering. The default is None.
        count_vectorizor : count vectorizor object, optional
            count vectorizor object that is used for ctf-idf. The default is None.
        ngram_range : tuple, optional
            range of ngrams to be considered. The default is (1,3).
        BERTkwargs : dict, optional
            dictionary of kwargs passed into bertopic. The default is {}.
        from_probs : boolean, optional
            true to assign topics to documents based on a probability threshold (i.e., documents can have multiple topics). 
            The default is True.
        thresh : float, optional
            probability threshold used when from_probs=True. The default is 0.01.

        Returns
        -------
        None.

        """

        self.sentence_models = {}; self.embeddings = {}; self.BERT_models = {}
        self.BERT_model_topics_per_doc = {}; self.BERT_model_probs={}; self.BERT_model_all_topics_per_doc={}
        for col in self.text_columns:
            if sentence_transformer_model:
                sentence_model = SentenceTransformer(sentence_transformer_model)
                corpus = self.data_df[col]
                embeddings = sentence_model.encode(corpus, show_progress_bar=False)
                topic_model = BERTopic(umap_model=umap, vectorizer_model=count_vectorizor, hdbscan_model=hdbscan,
                                       verbose=True, n_gram_range=ngram_range, embedding_model=sentence_model,
                                       calculate_probabilities=from_probs,
                                       **BERTkwargs)
                topics, probs = topic_model.fit_transform(corpus, embeddings)
                self.sentence_models[col] = sentence_model
                self.embeddings[col] = embeddings
            else:
                corpus = self.data_df[col]
                topic_model = BERTopic(umap_model=umap, vectorizer_model=count_vectorizor, hdbscan_model=hdbscan,
                                       verbose=True, n_gram_range=ngram_range, calculate_probabilities=from_probs,
                                       **BERTkwargs)
                topics, probs = topic_model.fit_transform(corpus)
            self.BERT_models[col] = topic_model
            self.BERT_model_topics_per_doc[col] = topics
            self.BERT_model_probs[col] = probs
            if from_probs == True:
                self.__get_all_topics_for_doc(col, probs, thresh, topics)
            self.reduced = False

    def __get_all_topics_for_doc(self, col, probs, thresh, topics):
        """
        Helper function that gets all the topics for every document based on a probability threshold
        
        Parameters
        ----------
        col : string
            column of the dataset currently being used
        probs : list
            list of probabilities per topic per document
        thresh : float
            threshold that a p should be greater than for a document to be considered in the topic
        topics : list
            list of topics
        """

        self.BERT_model_all_topics_per_doc[col] = [[] for i in range(len(probs))]
        for i in range(len(probs)):
            topic_probs = probs[i]#.strip("[]").split(" ")
            if len(topic_probs) > len(topics):
                topic_probs = [t for t in topic_probs if len(t)>0]
            topic_indices = [ind for ind in range(len(topic_probs)) if float(topic_probs[ind])>thresh]
            if len(topic_indices)==0:
                topic_indices = [-1]
                self.BERT_model_all_topics_per_doc[col][i] = [-1]
            else:
                self.BERT_model_all_topics_per_doc[col][i] = topic_indices
    
    def reduce_bert_topics(self, num=30, from_probs=False, thresh=0.01):
        """
        Reduces the number of topics in a trained bertopic model to the specified number.

        Parameters
        ----------
        num : int optional
            number of topics in the reduced model. The default is 30.
        from_probs : boolean, optional
            true to assign topics to documents based on a probability threshold (i.e., documents can have multiple topics). 
            The default is True.
        thresh : float, optional
            probability threshold used when from_probs=True. The default is 0.01.

        Returns
        -------
        None.

        """

        self.reduced = True
        for col in self.text_columns:
            corpus = self.data_df[col]
            topic_model = self.BERT_models[col]
            topic_model.reduce_topics(corpus, #self.BERT_model_topics_per_doc[col], 
                                                      #self.BERT_model_probs[col] , 
                                                      nr_topics=num)
            self.BERT_models[col] = topic_model
            self.BERT_model_topics_per_doc[col] = topic_model.topics_
            probs = topic_model.probabilities_
            if from_probs == True:
                self.__get_all_topics_for_doc(col, probs, thresh, self.BERT_model_topics_per_doc[col])
            self.BERT_model_probs[col] = probs
            
    def save_bert_topics(self, return_df=False, p_thres=0.0001, coherence=False, coh_method='u_mass', from_probs=False):
        """
        Saves bert topics results to file.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results dfs. The default is False.
        p_thres : float, optional
            word-topic probability threshold required for a word to be considered in a topic. 
            The default is 0.0001.
        coherence : boolean, optional
            true to calculate coherence for the model and save the results. The default is False.
        coh_method : string, optional
            Method used to calculate coherence. Can be any method used in gensim. 
            The default is 'u_mass'.
        from_probs : boolean, optional
            true to assign topics to documents based on a probability threshold (i.e., documents can have multiple topics). 
            The default is True.

        Returns
        -------
        dfs : dictionary of dataframes
            dictionary of dataframes where each key is a text column and each value is the corresponding
            topic model results

        """

        self.__create_folder()
        dfs = {}
        for col in self.text_columns:
            mdl = self.BERT_models[col]
            mdl_info = mdl.get_topic_info()
            doc_text = self.data_df[col].to_list()
            topics_data = {"topic number": [],           
                           "number of documents in topic": [],
                           "topic words": [],
                           "number of words": [],
                           "best documents": [],
                           "documents": []}
            if coherence == True: 
                self.get_bert_coherence(coh_method, from_probs=from_probs)
                topics_data["coherence"] = self.BERT_coherence[col]
            ordered_topics = list(set(mdl.topics_))
            ordered_topics.sort()
            for k in ordered_topics:
                topics_data["topic number"].append(k)
                topics_data["number of words"].append(len(mdl.get_topic(k)))
                topics_data["topic words"].append(", ".join([word[0] for word in mdl.get_topic(k) if word[1]>p_thres]))
                topics_data["number of documents in topic"].append(mdl_info.loc[mdl_info['Topic']==k].reset_index(drop=True).at[0,'Count'])
                if k!=-1: 
                    best_docs = [self.doc_ids[doc_text.index(text)] for text in mdl.get_representative_docs(k)]
                else:
                    best_docs = "n/a"
                topics_data["best documents"].append(best_docs)
                docs = [self.doc_ids[i] for i in range(len(self.BERT_model_topics_per_doc[col])) if self.BERT_model_topics_per_doc[col][i] == k]
                topics_data["documents"].append(docs)
            df = pd.DataFrame(topics_data)
            dfs[col] = df
            if return_df == False:
                if self.reduced:
                    file = os.path.join(self.folder_path,col+"_reduced_BERT_topics.csv")
                else: 
                    file = os.path.join(self.folder_path,col+"_BERT_topics.csv")
                df.to_csv(file)
        if return_df == True:
            return dfs
    
    def save_bert_topics_from_probs(self, thresh=0.01, return_df=False, coherence=False, coh_method='u_mass', from_probs=True):
        """
        Saves bertopic model results if using probability threshold.

        Parameters
        ----------
        thresh : float, optional
            probability threshold used when from_probs=True. The default is 0.01.
        return_df : boolean, optional
            True to return the results dfs. The default is False.
        coherence : boolean, optional
            true to calculate coherence for the model and save the results. The default is False.
        coh_method : string, optional
            Method used to calculate coherence. Can be any method used in gensim. 
            The default is 'u_mass'.
        from_probs : boolean, optional
            true to assign topics to documents based on a probability threshold (i.e., documents can have multiple topics). 
            The default is True.

        Returns
        -------
        topic_prob_dfs : dictionary of dataframes
            dictionary of dataframes where each key is a text column and each value is the corresponding
            topic model results

        """
        
        topic_dfs = self.save_bert_topics(return_df=True, coherence=coherence, coh_method=coh_method, from_probs=True)
        topic_prob_dfs = self.get_bert_topics_from_probs(topic_dfs, thresh, coherence)
        if return_df == False:
            for col in self.text_columns:
                if self.reduced:
                    file = os.path.join(self.folder_path,col+"_reduced_BERT_topics_modified.csv")
                else: 
                    file = os.path.join(self.folder_path,col+"_BERT_topics_modified.csv")
                topic_prob_dfs[col].to_csv(file)
        if return_df == True:
            return topic_prob_dfs
        
    def get_bert_topics_from_probs(self, topic_df, thresh=0.01, coherence=False):
        """
        Saves topic model results including each topic number, words, number of words,
        and best document when document topics are defined by a probabilty threshold.

        Parameters
        ----------
        topic_df : dictionary of dataframes
            dictionary of dataframes where each key is a text column and each value is the corresponding
            topic model results
        thresh : float, optional
            probability threshold used when from_probs=True. The default is 0.01.
        coherence : boolean, optional
            true to calculate coherence for the model and save the results. The default is False.
        
        Returns
        -------
        new_topic_dfs : dictionary of dataframes
            dictionary of dataframes where each key is a text column and each value is the corresponding
            topic model results

        """

        cols = ['topic number', 'topic words', 'number of words', 'best documents']
        if coherence == True: cols += ['coherence']
        new_topic_dfs = {col:topic_df[col][cols] for col in self.text_columns}
        for col in self.text_columns:
            documents_per_topic = {k:[] for k in new_topic_dfs[col]['topic number']}
            for i in range(len(self.BERT_model_all_topics_per_doc[col])): 
                doc_id = self.doc_ids[i]
                topics = self.BERT_model_all_topics_per_doc[col][i]
                for k in topics:
                    documents_per_topic[k].append(doc_id)
            num_docs = [len(docs) for docs in documents_per_topic.values()]
            new_topic_dfs[col]['documents'] = [docs for docs in documents_per_topic.values()]
            new_topic_dfs[col]['number of documents in topic'] = num_docs
        return new_topic_dfs
        
    def save_bert_taxonomy(self, return_df=False, p_thres=0.0001):
        """
        Saves a taxonomy of topics from bertopic model.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results dfs. The default is False.
        p_thres : float, optional
            word-topic probability threshold required for a word to be considered in a topic. 
            The default is 0.0001.

        Returns
        -------
        taxonomy_df : pandas Dataframe
            taxonomy dataframe with a column for each text column and each row a
            unique combination of topics found in the documents

        """

        self.__create_folder()
        taxonomy_data = {col:[] for col in self.text_columns}
        for col in self.text_columns:
            mdl = self.BERT_models[col]
            for doc in self.BERT_model_topics_per_doc[col]: 
                topic_num = doc
                words =  ", ".join([word[0] for word in mdl.get_topic(topic_num) if word[1]>p_thres])
                taxonomy_data[col].append(words)
        taxonomy_df = pd.DataFrame(taxonomy_data)
        taxonomy_df = taxonomy_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
            
        for i in range(len(taxonomy_df)):
            lesson_nums = []
            tax_row  = "\n".join([taxonomy_df.iloc[i][key] for key in taxonomy_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_data[key][j] for key in taxonomy_data])
                if doc_row == tax_row:                      
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_df["number of documents for row"] = num_lessons_per_row
        taxonomy_df = taxonomy_df.sort_values(by=[key for key in taxonomy_data])
        taxonomy_df = taxonomy_df.reset_index(drop=True)
        self.bert_taxonomy_df = taxonomy_df
        if return_df == True:
            return taxonomy_df
        if self.reduced:
            file = os.path.join(self.folder_path,"Reduced_BERT_taxonomy.csv")
        else: 
            file = os.path.join(self.folder_path,'BERT_taxonomy.csv')
        taxonomy_df.to_csv(file)
    
    def save_bert_document_topic_distribution(self, return_df=False):
        """
        Saves the document topic distribution.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results df. The default is False.
        
        Returns
        -------
        doc_df : pandas DataFrame
            dataframe with a row for each document and the probability for each topic 

        """

        self.__create_folder()
        doc_data = {col: [] for col in self.text_columns}
        doc_data['document number'] = self.doc_ids
        for col in self.text_columns:
            doc_data[col] = [l for l in self.BERT_model_probs[col]]
        doc_df = pd.DataFrame(doc_data)#{key:pd.Series(value) for key, value in doc_data.items()})
        if return_df == True:
            return doc_df
        if self.reduced:
            file = os.path.join(self.folder_path,"Reduced_BERT_topic_dist_per_doc.csv")
        else: 
            file = os.path.join(self.folder_path,"BERT_topic_dist_per_doc.csv")
        doc_df.to_csv(file)
        
    def save_bert_results(self, coherence=False, coh_method='u_mass', from_probs=False, thresh=0.01, topk=10):
        """
        Saves the taxonomy, coherence, and document topic distribution in one excel file.

        Parameters
        ----------
        coherence : boolean, optional
            true to calculate coherence for the model and save the results. The default is False.
        coh_method : string, optional
            Method used to calculate coherence. Can be any method used in gensim. 
            The default is 'u_mass'.
        from_probs : boolean, optional
            true to assign topics to documents based on a probability threshold (i.e., documents can have multiple topics). 
            The default is False.
        thresh : float, optional
            probability threshold used when from_probs=True. The default is 0.01.
        topk : int, optional
            Number of words per topic used to calculate diversity. The default is 10.
        
        Returns
        -------
        None.

        """
        
        self.__create_folder()
        data = {}
        if from_probs == False:
            topics_dict = self.save_bert_topics(return_df=True, coherence=coherence, coh_method=coh_method)
        elif from_probs == True:
            topics_dict = self.save_bert_topics_from_probs(thresh=thresh, return_df=True, coherence=coherence, coh_method=coh_method)
        data.update(topics_dict)
        data["taxonomy"] = self.save_bert_taxonomy(return_df=True)
        if coherence == True: data["coherence"] = self.save_bert_coherence(return_df=True, coh_method=coh_method, from_probs=from_probs)
        data["document topic distribution"] = self.save_bert_document_topic_distribution(return_df=True)
        data["topic diversity"] = self.save_bert_topic_diversity(topk=topk, return_df=True)
        if self.reduced:
            file = os.path.join(self.folder_path,"Reduced_BERTopic_results.xlsx")
        else: 
            file = os.path.join(self.folder_path,'BERTopic_results.xlsx')
        with pd.ExcelWriter(file) as writer2:
            for results in data:
                if len(results) >31:
                    result_label = results[:31]
                else:
                    result_label = results
                data[results].to_excel(writer2, sheet_name = result_label, index = False)
    
    def save_bert_vis(self):
        """
        Saves the bertopic visualization and hierarchy visualization.

        Returns
        -------
        None.

        """

        self.__create_folder()
        if self.reduced:
            file = os.path.join(self.folder_path, 'Reduced')
        else: 
            file = os.path.join(self.folder_path,"")
        for col in self.text_columns:
            topic_model = self.BERT_models[col]
            fig = topic_model.visualize_topics()
            fig.write_html(file+'bertopics_viz.html')
            hfig = topic_model.visualize_hierarchy()
            hfig.write_html(file+'bertopics_hierarchy_viz.html')
    
    def hdp(self, training_iterations=1000, iteration_step=10, to_lda=True, kwargs={}, topic_threshold=0.0):
        """
        Performs HDP topic modeling which is useful when the number of topics is not known.

        Parameters
        ----------
        training_iterations : int, optional
            number of training iterations. The default is 1000.
        iteration_step : int, optional
            number of steps per iteration. The default is 10.
        to_lda : boolean, optional
            True to convert the hdp model to an lda model. The default is True.
        kwargs : dict, optional
            kwargs to pass into the hdp model. The default is {}.
        topic_threshold : float, optional
            probability threshold used when converting hdp topics to lda topics. The default is 0.0.

        Returns
        -------
        None.

        """

        start = time()
        self.hdp_models = {}
        self.hdp_coherence = {}
        for col in self.text_columns:
            texts = self.data_df[col].tolist()
            if self.ngrams == "tp":
                corpus = self.__create_corpus_of_ngrams(texts)
                hdp = tp.HDPModel(tw = tp.TermWeight.IDF, corpus=corpus, **kwargs)
            else:
                hdp = tp.HDPModel(tw = tp.TermWeight.IDF, **kwargs)
                for text in texts:
                    hdp.add_doc(text)
            sleep(0.5)
            for i in tqdm(range(0, training_iterations, iteration_step), col+" HDP…"):
                hdp.train(iteration_step)
            self.hdp_models[col] = hdp
            #self.hdp_coherence[col] = self.coherence_scores(hdp, "lda")
        if to_lda:
            self.lda_models = {}
            self.lda_coherence = {}
            self.lda_num_topics = {}
            for col in self.text_columns:
                self.lda_models[col], new_topic_ids = self.hdp_models[col].convert_to_lda(topic_threshold=topic_threshold)
                self.lda_num_topics[col] = self.lda_models[col].k
                self.lda_coherence[col] = self.coherence_scores(self.lda_models[col], "lda")
        print("HDP: ", (time()-start)/60, " minutes")
        
    def coherence_scores(self, mdl, lda_or_hlda, measure='c_v'):
        """
        Computes and returns coherence scores for lda and hlda models.

        Parameters
        ----------
        mdl : lda or hlda model object
            topic model object created previously
        lda_or_hlda : str
            denotes whether coherence is being calculated for lda or hlda
        measure : string, optional
            denotes which coherence metric to compute. The default is 'c_v'.

        Returns
        -------
        scores : dict
            coherence scores, averages, and std dev

        """
        
        scores = {}
        coh = tp.coherence.Coherence(mdl, coherence= measure)
        if lda_or_hlda == "hlda":
            scores["per topic"] = [coh.get_score(topic_id=k) for k in range(mdl.k) if (mdl.is_live_topic(k) and mdl.num_docs_of_topic(k)>0)]
            for level in range(1, self.levels):
                level_scores = []
                for k in range(mdl.k):
                    if int(mdl.level(k)) == level:
                        level_scores.append(coh.get_score(topic_id=k))
                scores["level "+str(level)+" average"] = np.average(level_scores)
                scores["level "+str(level)+" std dev"] = np.std(level_scores)
        elif lda_or_hlda == "lda":
            scores["per topic"] = [coh.get_score(topic_id=k) for k in range(mdl.k)]
        scores["average"] = np.average(scores["per topic"])
        scores['std dev'] = np.std(scores["per topic"])
        return scores
    
    def __create_corpus_of_ngrams(self, texts):
        """
        Creates ngrams for a corpus using tomotopy.

        Parameters
        ----------
        texts : list
            list of documents in string format.

        Returns
        -------
        corpus : tomotopy corpus object
            tomotopy corpus object with ngrams

        """

        corpus = tp.utils.Corpus()
        for text in texts:
            corpus.add_doc(text)
        #identifies n_grams
        cands = corpus.extract_ngrams(min_cf=1, min_df=1, max_len=3)
        #transforms corpus to contain n_grams
        corpus.concat_ngrams(cands, delimiter=' ')
        return corpus
    
    def __find_optimized_lda_topic_num(self, col, max_topics, training_iterations=1000, iteration_step=10, coh_thres = 0.005, **kwargs):
        """ 
        Under development. Used to automate the elbow method for finding the ideal
        number of topics in lda models

        Parameters
        ----------
        col : string
            text column of the dataset the topic optimization is performed on
        max_topics : int
            maximum number of topics to consider
        training_iterations : int, optional
            number of training iterations. The default is 1000.
        iteration_step : int, optional
            number of steps per iteration. The default is 10.
        coh_thres : float, optional
            the threshold for the difference in coherence needed to define the best number of topics.
            If the difference in coherence between a model with topics = k+1 and a model with
            topics = k is less than the threshold, then k+1 is the ideal topic number.
            The default is 0.005.
        **kwargs : dict
            any kwargs for the lda topic model.

        Returns
        -------
        None.

        """

        coherence = []
        LL = []
        perplexity = []
        topic_num = [i for i in range(1, max_topics+1)]
        ##need to address this specifically what percentage is removed
        texts = self.data_df[col].tolist()
        sleep(0.5)
        for num in tqdm(topic_num, col+" LDA optimization…"):
            if self.ngrams == "tp":
                corpus = self.__create_corpus_of_ngrams(texts)
                lda = tp.LDAModel(k=num, tw = tp.TermWeight.IDF, corpus=corpus, **kwargs)
            else:
                lda = tp.LDAModel(k=num, tw = tp.TermWeight.IDF, **kwargs)
                for text in texts:
                    lda.add_doc(text)
            sleep(0.5)
            for i in range(0, training_iterations, iteration_step):
                lda.train(iteration_step)
            coherence.append(self.coherence_scores(lda, "lda")["average"])
            LL.append(lda.ll_per_word)
            perplexity.append(lda.perplexity)
        #print(coherence, perplexity)
        coherence = normalize(np.array([coherence,np.zeros(len(coherence))]))[0]
        perplexity = normalize(np.array([perplexity,np.zeros(len(perplexity))]))[0]
        #plots optomization graph
        plt.figure()
        plt.xlabel("Number of Topics")
        plt.ylabel("Normalized Score")
        plt.title("LDA optimization for "+col)
        plt.plot(topic_num, coherence, label="Coherence (c_v)", color="purple")
        plt.plot(topic_num, perplexity, label="Perplexity", color="green")
        plt.legend()
        plt.show()
        self.__create_folder()
        plt.savefig(os.path.join(self.folder_path,"LDA_optimization_"+col+"_.png"))
        plt.close()
#        plt.figure()
#        plt.xlabel("Number of Topics")
#        plt.ylabel("Perplexity")
#        plt.title("LDA optimization for "+col)
#        plt.plot(topic_num, perplexity, marker='o', color="green")
#        plt.show()
#        self.__create_folder()
#        plt.savefig(self.folder_path+"/LDA_optimization_P_"+col+"_.png")
#
#        plt.close()
#        plt.figure()
#        plt.xlabel("Number of Topics")
#        plt.ylabel("Loglikelihood")
#        plt.title("LDA optimization for "+col)
#        plt.plot(topic_num, LL, marker='o', color="blue")
#        plt.show()
#        self.__create_folder()
#        plt.savefig(self.folder_path+"/LDA_optimization_LL_"+col+"_.png")
        #want to minimize perplexity, maximize coherence, look for max difference between the two
        #diff = [coherence[i]-perplexity[i] for i in range(len(topic_num))]
        #change_in_diff = [abs(diff[i]-diff[i+1])-abs(diff[i+1]-diff[i+2]) for i in range(0, len(diff)-2)]
        
        best_index = 0
        diffs = [abs(coherence[i]-coherence[i-1]) for i in range(1, len(coherence))]
        for diff in diffs:
            if diff<coh_thres:
                best_index = diffs.index(diff)
                break
        best_num_of_topics = topic_num[best_index]
        self.lda_num_topics[col] = best_num_of_topics
        
    def __lda_optimization(self, max_topics=200,training_iterations=1000, iteration_step=10, thres = 0.005, **kwargs):
        """
        Runs the lda optimization for all text columns.

        Parameters
        ----------
        max_topics : int
            maximum number of topics to consider
        training_iterations : int, optional
            number of training iterations. The default is 1000.
        iteration_step : int, optional
            number of steps per iteration. The default is 10.
        coh_thres : float, optional
            the threshold for the difference in coherence needed to define the best number of topics.
            If the difference in coherence between a model with topics = k+1 and a model with
            topics = k is less than the threshold, then k+1 is the ideal topic number.
            The default is 0.005.
        **kwargs : dict
            any kwargs for the lda topic model.

        Returns
        -------
        None.

        """
        #needs work
        start = time()
        self.lda_num_topics = {}
        for col in self.text_columns:
            self.__find_optimized_lda_topic_num(col, max_topics, training_iterations=1000, iteration_step=10, thres = 0.005, **kwargs)
            #print(self.lda_num_topics[col], " topics for ", col)
        print("LDA topic optomization: ", (time()-start)/60, " minutes")
    
    def lda(self, num_topics={}, training_iterations=1000, iteration_step=10, max_topics=0, **kwargs):
        """
        Performs LDA topic modeling.

        Parameters
        ----------
        num_topics : dict, optional
            keys are columns in text_columns, values are the number of topics lda forms
            optional - if omitted, lda optimization is run and produces the num_topics The default is {}.
        training_iterations : int, optional
            number of training iterations. The default is 1000.
        iteration_step : int, optional
            number of steps per iteration. The default is 10.
        max_topics : int, optional
            maximum number of topics to consider The default is 0.
        **kwargs : dict
            any kwargs for the lda topic model.

        Returns
        -------
        None.

        """

        start = time()
        self.lda_models = {}
        self.lda_coherence = {}
        if num_topics == {}:
            if max_topics == 0:
                max_topics = 200
            self.__lda_optimization(max_topics=max_topics, **kwargs)
        else:
            self.lda_num_topics = num_topics
        for col in self.text_columns:
            texts = self.data_df[col].tolist()
            if self.ngrams == "tp":
                corpus = self.__create_corpus_of_ngrams(texts)
                lda = tp.LDAModel(k=self.lda_num_topics[col], tw = tp.TermWeight.IDF, corpus=corpus, **kwargs)
            else:
                lda = tp.LDAModel(k=self.lda_num_topics[col], tw = tp.TermWeight.IDF, **kwargs)
                for text in texts:
                    lda.add_doc(text)
            sleep(0.5)
            for i in tqdm(range(0, training_iterations, iteration_step), col+" LDA…"):
                lda.train(iteration_step)
            self.lda_models[col] = lda
            self.lda_coherence[col] = self.coherence_scores(lda, "lda")
        print("LDA: ", (time()-start)/60, " minutes")
        
    def save_lda_models(self):
        """
        Saves lda models to file.

        Returns
        -------
        None.

        """

        self.__create_folder()
        for col in self.text_columns:
            mdl = self.lda_models[col]
            mdl.save(os.path.join(self.folder_path,col+"_lda_model_object.bin"))
        self.data.save(results_path=os.path.join(self.folder_path,"preprocessed_data.csv"))
    
    def save_lda_document_topic_distribution(self, return_df=False):
        """
        Saves lda document topic distribution to file or returns the dataframe to another function.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results df. The default is False.
        
        Returns
        -------
        doc_df : pandas DataFrame
            dataframe with a row for each document and the probability for each topic

        """
        
        #identical to hlda function except for lda tag
        self.__create_folder()
        doc_data = {col: [] for col in self.text_columns}
        doc_data['document number'] = self.doc_ids
        for col in self.text_columns:
            mdl = self.lda_models[col]
            for doc in mdl.docs:
                doc_data[col].append(doc.get_topic_dist())
        doc_df = pd.DataFrame(doc_data)
        if return_df == True:
            return doc_df
        doc_df.to_csv(os.path.join(self.folder_path,"lda_topic_dist_per_doc.csv"))
        #print("LDA topic distribution per document saved to: ",self.folder_path+"/lda_topic_dist_per_doc.csv")
    
    def save_lda_coherence(self, return_df=False):
        """
        Saves lda coherence to file or returns the dataframe to another function.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results df. The default is False.

        Returns
        -------
        coherence_df : pandas DataFrame
            Dataframe with each row a topic, column has coherence scores.

        """
        
        self.__create_folder()
        max_topics = max([value for value in self.lda_num_topics.values()])
        coherence_score = {"topic numbers": ["average score"]+['std dev']+[i for i in range(0,max_topics)]}
        for col in self.text_columns:
            coherence_score[col] = []
            c_scores = self.lda_coherence[col]
            average_coherence = c_scores['average']
            coherence_score[col].append(average_coherence)
            std_coherence = c_scores['std dev']
            coherence_score[col].append(std_coherence)
            coherence_per_topic = c_scores['per topic']
            for i in range(0, (max_topics-len(coherence_per_topic))):
                coherence_per_topic.append("n/a")
            coherence_score[col] += coherence_per_topic
        coherence_df = pd.DataFrame(coherence_score)
        if return_df == True:
            return coherence_df
        coherence_df.to_csv(os.path.join(self.folder_path,"lda_coherence.csv"))
    
    def save_lda_topics(self, return_df=False, p_thres=0.001):
        """
        Saves lda topics to file.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results df. The default is False.
        p_thres : float, optional
            word-topic probability threshold required for a word to be considered in a topic. 
            The default is 0.001.

        Returns
        -------
        dfs : dictionary of dataframes
            dictionary of dataframes where each key is a text column and each value is the corresponding
            topic model results

        """
        
        #saving raw topics with coherence
        self.__create_folder()
        dfs = {}
        for col in self.text_columns:
            mdl = self.lda_models[col]
            topics_data = {"topic number": [],           
                           "number of documents in topic": [],
                           "topic words": [],
                           "total number of words": [],
                           "number of words": [],
                           "best document": [],
                           "coherence": [],
                           "documents": []}
            topics_data["coherence"] = self.lda_coherence[col]["per topic"]
            for k in range(mdl.k):
                topics_data["topic number"].append(k)
                topics_data["total number of words"].append(mdl.get_count_by_topics()[k])
                probs = mdl.get_topic_word_dist(k)
                probs = [p for p in probs if p>p_thres]
                topics_data["number of words"].append(len(probs))
                topics_data["topic words"].append(", ".join([word[0] for word in mdl.get_topic_words(k, top_n=len(probs))]))
            docs_in_topic ={k:[] for k in range(mdl.k)}
            probs = {k:[] for k in range(mdl.k)}
            i = 0
            for doc in mdl.docs:
                for topic, weight in doc.get_topics(top_n=5):
                    docs_in_topic[topic].append(self.doc_ids[i])
                    probs[topic].append(weight)
                i+=1
            #print(probs)
            for k in docs_in_topic:
                topics_data["best document"].append(docs_in_topic[k][probs[k].index(max(probs[k]))])
                topics_data["number of documents in topic"].append(len(docs_in_topic[k]))
                topics_data["documents"].append(docs_in_topic[k])
            df = pd.DataFrame(topics_data)
            dfs[col] = df
            if return_df == False:
                df.to_csv(os.path.join(self.folder_path,col+"_lda_topics.csv"))
                #print("LDA topics for "+col+" saved to: ",self.folder_path+"/"+col+"_lda_topics.csv")
        if return_df == True:
            return dfs
    
    def save_lda_taxonomy(self, return_df=False, use_labels=False, num_words=10):
        """
        Saves lda taxonomy to file or returns the dataframe to another function.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results df. The default is False.
        use_labels : boolean, optional
            True to use topic labels generated from tomotopy. The default is False.
        num_words : int, optional
            Number of words to display in the taxonomy. The default is 10.

        Returns
        -------
        taxonomy_df : pandas Dataframe
            Taxonomy dataframe with a column for each text column and each row a
            unique combination of topics found in the documents

        """
        
        self.__create_folder()
        taxonomy_data = {col:[] for col in self.text_columns}
        for col in self.text_columns:
            mdl = self.lda_models[col]
            for doc in mdl.docs: 
                topic_num = int(doc.get_topics(top_n=1)[0][0])
                if use_labels == False:
                    num_words = min(mdl.get_count_by_topics()[topic_num], num_words)
                    words =  ", ".join([word[0] for word in mdl.get_topic_words(topic_num, top_n=num_words)])
                else:
                    words = ", ".join(self.lda_labels[col][topic_num])
                #if len(words) > 35000:
                #    words = words[0:words.rfind(", ")]
                taxonomy_data[col].append(words)
        taxonomy_df = pd.DataFrame(taxonomy_data)
        taxonomy_df = taxonomy_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
        for i in range(len(taxonomy_df)):
            lesson_nums = []
            tax_row  = "\n".join([taxonomy_df.iloc[i][key] for key in taxonomy_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_data[key][j] for key in taxonomy_data])
                if doc_row == tax_row:                      
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_df["number of documents for row"] = num_lessons_per_row
        taxonomy_df = taxonomy_df.sort_values(by=[key for key in taxonomy_data])
        taxonomy_df = taxonomy_df.reset_index(drop=True)
        self.lda_taxonomy_df = taxonomy_df
        if return_df == True:
            return taxonomy_df
        taxonomy_df.to_csv(os.path.join(self.folder_path,'lda_taxonomy.csv'))
        #print("LDA taxonomy saved to: ", os.path.join(self.folder_path,'lda_taxonomy.csv'))
    
    def save_lda_results(self):
        """
        Saves the taxonomy, coherence, and document topic distribution in one excel file.

        Returns
        -------
        None.

        """

        self.__create_folder()
        data = {}
        topics_dict = self.save_lda_topics(return_df=True)
        data.update(topics_dict)
        data["taxonomy"] = self.save_lda_taxonomy(return_df=True)
        data["coherence"] = self.save_lda_coherence(return_df=True)
        data["document topic distribution"] = self.save_lda_document_topic_distribution(return_df=True)
        with pd.ExcelWriter(os.path.join(self.folder_path,'lda_results.xlsx')) as writer2:
            for results in data:
                data[results].to_excel(writer2, sheet_name = results, index = False)
        #print("LDA results saved to: ", os.path.join(self.folder_path,'lda_results.xlsx'))
        
    def lda_extract_models(self, file_path):
        """
        Loads lda models from file.

        Parameters
        ----------
        file_path : str
            path to file

        Returns
        -------
        None.

        """

        self.lda_num_topics = {}
        self.lda_coherence = {}
        self.lda_models = {}
        for col in self.text_columns:
            self.lda_models[col] = tp.LDAModel.load(os.path.join(file_path,col+"_lda_model_object.bin"))
            self.lda_coherence[col] = self.coherence_scores(self.lda_models[col], "lda")
            self.lda_num_topics[col] = self.lda_models[col].k
        #print("LDA models extracted from: ", file_path)
        preprocessed_filepath = os.path.join(file_path,"preprocessed_data")
        self.data.load(preprocessed_filepath+".csv", preprocessed=True, id_col=self.data.id_col, text_columns=self.data.text_columns)
        self.folder_path = file_path
        
    def lda_visual(self, col):
        """
        Saves pyLDAvis output from lda to file.

        Parameters
        ----------
        col : str
            reference to column of interest

        Returns
        -------
        None.

        """

        self.__create_folder()
        mdl = self.lda_models[col]
        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
        doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq
        prepared_data = pyLDAvis.prepare(
            topic_term_dists, 
            doc_topic_dists, 
            doc_lengths, 
            vocab, 
            term_frequency
        )
        pyLDAvis.save_html(prepared_data, os.path.join(self.folder_path,col+'_ldavis.html'))
        #print("LDA Visualization for "+col+" saved to: "+self.folder_path+'/'+col+'_ldavis.html')
    
    def hlda_visual(self, col):
        """
        Saves pyLDAvis output from hlda to file.

        Parameters
        ----------
        col : str
            reference to column of interest

        Returns
        -------
        None.

        """

        self.__create_folder()
        mdl = self.hlda_models[col]
        topics = [k for k in range(mdl.k) if mdl.is_live_topic(k)]
        
        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k) if mdl.is_live_topic(k)])
        doc_topic_dists_pre_stack = []
        for doc in mdl.docs:
            doc_topics = []
            for k in topics:
                if k in doc.path:
                    doc_topics.append(list(doc.path).index(k))
                else:
                    doc_topics.append(0)
            doc_topic_dists_pre_stack.append(doc_topics)
        doc_topic_dists = np.stack(doc_topic_dists_pre_stack)
        doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq
        prepared_data = pyLDAvis.prepare(
            topic_term_dists, 
            doc_topic_dists, 
            doc_lengths, 
            vocab, 
            term_frequency
        )
        print(os.path.join(self.folder_path,col+'_hldavis.html'))
        pyLDAvis.save_html(prepared_data, os.path.join(self.folder_path,col+'_hldavis.html'))
        #print("hLDA Visualization for "+col+" saved to: "+self.folder_path+'/'+col+'_hldavis.html')
    
    def label_lda_topics(self, extractor_min_cf=5, extractor_min_df=3, extractor_max_len=5, extractor_max_cand=5000, labeler_min_df=5, labeler_smoothing=1e-2, labeler_mu=0.25, label_top_n=3):
        """
        Uses tomotopy's auto topic labeling tool to label topics. Stores labels in class; after running this function, a flag can be used to use labels or not in taxonomy saving functions.
        
        ARGUMENTS
        ---------
        extractor_min_cf : int
            from tomotopy docs: "minimum collection frequency of collocations. Collocations with a smaller collection frequency than min_cf are excluded from the candidates. Set this value large if the corpus is big"
        extractor_min_df : int
            from tomotopy docs: "minimum document frequency of collocations. Collocations with a smaller document frequency than min_df are excluded from the candidates. Set this value large if the corpus is big"
        extractor_max_len : int
            from tomotopy docs: "maximum length of collocations"
        extractor_max_cand : int
            from tomotopy docs: "maximum number of candidates to extract"
        labeler_min_df : int
            from tomotopy docs: "minimum document frequency of collocations. Collocations with a smaller document frequency than min_df are excluded from the candidates. Set this value large if the corpus is big"
        labeler_smoothing : float
            from tomotopy docs: "a small value greater than 0 for Laplace smoothing"
        labeler_mu : float
            from tomotopy docs: "a discriminative coefficient. Candidates with high score on a specific topic and with low score on other topics get the higher final score when this value is the larger."
        label_top_n : int
            from tomotopy docs: "the number of labels"
        """
        
        extractor = tp.label.PMIExtractor(min_cf=extractor_min_cf, min_df=extractor_min_df, max_len=extractor_max_len, max_cand=extractor_max_cand)
        
        self.lda_labels = {}
        for col in self.text_columns:
            cands = extractor.extract(self.lda_models[col])
            labeler = tp.label.FoRelevance(self.lda_models[col], cands, min_df=labeler_min_df, smoothing=labeler_smoothing, mu=labeler_mu)
            self.lda_labels[col] = []
            for k in range(self.lda_models[col].k):
                label_w_probs = labeler.get_topic_labels(k,top_n=label_top_n)
                label = [word for word,prob in label_w_probs]
                self.lda_labels[col].append(label)
        
    def label_hlda_topics(self, extractor_min_cf=5, extractor_min_df=3, extractor_max_len=5, extractor_max_cand=5000, labeler_min_df=5, labeler_smoothing=1e-2, labeler_mu=0.25, label_top_n=3):
        """
        Uses tomotopy's auto topic labeling tool to label topics. Stores labels in class; after running this function, a flag can be used to use labels or not in taxonomy saving functions.
        
        ARGUMENTS
        ---------
        extractor_min_cf : int
            from tomotopy docs: "minimum collection frequency of collocations. Collocations with a smaller collection frequency than min_cf are excluded from the candidates. Set this value large if the corpus is big"
        extractor_min_df : int
            from tomotopy docs: "minimum document frequency of collocations. Collocations with a smaller document frequency than min_df are excluded from the candidates. Set this value large if the corpus is big"
        extractor_max_len : int
            from tomotopy docs: "maximum length of collocations"
        extractor_max_cand : int
            from tomotopy docs: "maximum number of candidates to extract"
        labeler_min_df : int
            from tomotopy docs: "minimum document frequency of collocations. Collocations with a smaller document frequency than min_df are excluded from the candidates. Set this value large if the corpus is big"
        labeler_smoothing : float
            from tomotopy docs: "a small value greater than 0 for Laplace smoothing"
        labeler_mu : float
            from tomotopy docs: "a discriminative coefficient. Candidates with high score on a specific topic and with low score on other topics get the higher final score when this value is the larger."
        label_top_n : int
            from tomotopy docs: "the number of labels"
        """
        
        extractor = tp.label.PMIExtractor(min_cf=extractor_min_cf, min_df=extractor_min_df, max_len=extractor_max_len, max_cand=extractor_max_cand)
        
        self.hlda_labels = {}
        for col in self.text_columns:
            cands = extractor.extract(self.hlda_models[col])
            labeler = tp.label.FoRelevance(self.lda_models[col], cands, min_df=labeler_min_df, smoothing=labeler_smoothing, mu=labeler_mu)
            self.hlda_labels[col] = []
            for k in range(self.hlda_models[col].k):
                label_w_probs = labeler.get_topic_labels(k,top_n=label_top_n)
                label = [word for word,prob in label_w_probs]
                self.hlda_labels[col].append(label)
    
    def save_mixed_taxonomy(self,use_labels=False):
        """
        A custom mixed lda/hlda model taxonomy. Must run lda and hlda with desired parameters first.

        Parameters
        ----------
        use_labels : boolean, optional
            True to use topic labels. The default is False.

        Returns
        -------
        None.

        """

        col_for_lda = [self.text_columns[i] for i in [0,2]]
        col_for_hlda = self.text_columns[1]
        
        self.__create_folder()
        taxonomy_data = {'Lesson(s) Learned':[],'Driving Event Level 1':[],'Driving Event Level 2':[],'Recommendation(s)':[]}

        # first column; lda
        col = self.text_columns[0]
        mdl = self.lda_models[col]
        for doc in mdl.docs:
            topic_num = int(doc.get_topics(top_n=1)[0][0])
            if use_labels == False:
                num_words = min(mdl.get_count_by_topics()[topic_num], 100)
                words =  ", ".join([word[0] for word in mdl.get_topic_words(topic_num, top_n=num_words)])
            else:
                words = ", ".join(self.lda_labels[col][topic_num])
            taxonomy_data[col].append(words)
        
        # second column; hlda
        col = self.text_columns[1]
        mdl = self.hlda_models[col]
        for doc in mdl.docs:
            topic_nums = doc.path
            for level in range(1, self.levels):
                if use_labels == False:
                    words =  ", ".join([word[0] for word in mdl.get_topic_words(topic_nums[level], top_n=500)])
                else:
                    words = ", ".join(self.hlda_labels[col][topic_nums[level]])
                taxonomy_data[col+" Level "+str(level)].append(words)
                
        # third column; lda
        col = self.text_columns[2]
        mdl = self.lda_models[col]
        for doc in mdl.docs:
            topic_num = int(doc.get_topics(top_n=1)[0][0])
            if use_labels == False:
                num_words = min(mdl.get_count_by_topics()[topic_num], 100)
                words =  ", ".join([word[0] for word in mdl.get_topic_words(topic_num, top_n=num_words)])
            else:
                words = ", ".join(self.lda_labels[col][topic_num])
            taxonomy_data[col].append(words)
        
        self.taxonomy_data = taxonomy_data
        taxonomy_df = pd.DataFrame(taxonomy_data)
        taxonomy_df = taxonomy_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
        for i in range(len(taxonomy_df)):
            lesson_nums = []
            tax_row  = "\n".join([taxonomy_df.iloc[i][key] for key in taxonomy_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_data[key][j] for key in taxonomy_data])
                if doc_row == tax_row:
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_df["number of documents for row"] = num_lessons_per_row
        taxonomy_df = taxonomy_df.sort_values(by=[key for key in taxonomy_data])
        taxonomy_df = taxonomy_df.reset_index(drop=True)
        self.taxonomy_df = taxonomy_df
        taxonomy_df.to_csv(os.path.join(self.folder_path,'mixed_taxonomy.csv'))
    
    def hlda(self, levels=3, training_iterations=1000, iteration_step=10, **kwargs):
        """
        Performs hlda topic modeling.

        Parameters
        ----------
        levels : int, optional
            number of hierarchical levels. The default is 3.
        training_iterations : int, optional
            number of training iterations. The default is 1000.
        iteration_step : int, optional
            number of steps per iteration. The default is 10.
        **kwargs : dict
            any kwargs for the hlda topic model

        Returns
        -------
        None.

        """

        start = time()
        self.hlda_models = {}
        self.hlda_coherence = {}
        self.levels = levels
        for col in self.text_columns:
            texts = self.data_df[col].tolist()
            if self.ngrams == "tp":
                corpus = self.__create_corpus_of_ngrams(texts)
                mdl = tp.HLDAModel(depth=levels, tw = tp.TermWeight.IDF, corpus=corpus, **kwargs)
            else: 
                mdl = tp.HLDAModel(depth=levels, tw = tp.TermWeight.IDF, **kwargs)
                for text in texts:
                    mdl.add_doc(text)
            sleep(0.5)
            for i in tqdm(range(0, training_iterations, iteration_step), col+" hLDA…"):
                mdl.train(iteration_step)
                self.hlda_models[col]=mdl
                sleep(0.5)
            self.hlda_coherence[col] = self.coherence_scores(mdl, "hlda")
            sleep(0.5)
        print("hLDA: ", (time()-start)/60, " minutes")
        return
    
    def save_hlda_document_topic_distribution(self, return_df=False):
        """
        Saves hlda document topic distribution to file.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results df. The default is False.
        
        Returns
        -------
        doc_df : pandas DataFrame
            dataframe with a row for each document and the probability for each topic

        """
        
        self.__create_folder()
        doc_data = {col: [] for col in self.text_columns}
        doc_data['document number']=self.doc_ids
        for col in self.text_columns:
            mdl = self.hlda_models[col]
            for doc in mdl.docs:
                doc_data[col].append(doc.get_topic_dist())
        doc_df = pd.DataFrame(doc_data)
        if return_df == True:
            return doc_df
        doc_df.to_csv(os.path.join(self.folder_path,'hlda_topic_dist_per_doc.csv'))
        #print("hLDA topic distribution per document saved to: ",self.folder_path+"hlda_topic_dist_per_doc.csv")
    
    def save_hlda_models(self):
        """
        Saves hlda models to file.
        
        Returns
        -------
        None.

        """

        self.__create_folder()
        for col in self.text_columns:
            mdl = self.hlda_models[col]
            mdl.save(os.path.join(self.folder_path,col+"_hlda_model_object.bin"))
            #print("hLDA model for "+col+" saved to: ", (self.folder_path+"/"+col+"_hlda_model_object.bin"))
        self.data.save(results_path=os.path.join(self.folder_path,"preprocessed_data.csv"))
        
    def save_hlda_topics(self, return_df=False, p_thres=0.001):
        """ 
        Saves hlda topics to file.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results df. The default is False.
        p_thres : float, optional
            word-topic probability threshold required for a word to be considered in a topic. 
            The default is 0.001.

        Returns
        -------
        dfs : dictionary of dataframes
            dictionary of dataframes where each key is a text column and each value is the corresponding
            topic model results

        """

        #saving raw topics with coherence
        self.__create_folder()
        dfs = {}
        for col in self.text_columns:
            mdl = self.hlda_models[col]
            topics_data = {"topic level": [],
                "topic number": [],
                "parent": [],
                "number of documents in topic": [],
                "topic words": [],
                "total number of words": [],
                "number of words": [],
                "best document": [],
                "coherence": [],
                "documents": []}
            topics_data["coherence"] = self.hlda_coherence[col]["per topic"]
            for k in range(mdl.k):
                if not mdl.is_live_topic(k) or mdl.num_docs_of_topic(k)<0:
                    continue
                topics_data["parent"].append(mdl.parent_topic(k))
                topics_data["topic level"].append(mdl.level(k))
                topics_data["number of documents in topic"].append(mdl.num_docs_of_topic(k))
                topics_data["topic number"].append(k)
                probs = mdl.get_topic_word_dist(k)
                probs = [p for p in probs if p>p_thres]
                topics_data["number of words"].append(len(probs))
                topics_data["total number of words"].append(mdl.get_count_by_topics()[k])
                topics_data["topic words"].append(", ".join([word[0] for word in mdl.get_topic_words(k, top_n=len(probs))]))
                i = 0
                docs_in_topic = []
                probs = []
                for doc in mdl.docs:
                    if doc.path[mdl.level(k)] == k:
                        prob = doc.get_topic_dist()[mdl.level(k)]
                        docs_in_topic.append(self.doc_ids[i])
                        probs.append(prob)
                    i += 1
                topics_data["best document"].append(docs_in_topic[probs.index(max(probs))])
                topics_data["documents"].append(docs_in_topic)
            df = pd.DataFrame(topics_data)
            dfs[col] = df
            if return_df == False:
                df.to_csv(os.path.join(self.folder_path,col+"_hlda_topics.csv"))
                #print("hLDA topics for "+col+" saved to: ",self.folder_path+"/"+col+"_hlda_topics.csv")
        if return_df == True:
            return dfs
            
    def save_hlda_coherence(self, return_df=False):
        """
        Saves hlda coherence to file.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results df. The default is False.

        Returns
        -------
        coherence_df : pandas DataFrame
            Dataframe with each row a topic, column has coherence scores.

        """

        self.__create_folder()
        coherence_data = {}
        for col in self.text_columns:
            coherence_data[col+" average"]=[]; coherence_data[col+" std dev"]=[]
            for level in range(self.levels):
                if level == 0:
                    coherence_data[col+" average"].append(self.hlda_coherence[col]["average"])
                    coherence_data[col+" std dev"].append(self.hlda_coherence[col]["std dev"])
                else:
                    coherence_data[col+" std dev"].append(self.hlda_coherence[col]["level "+str(level)+" std dev"])
                    coherence_data[col+" average"].append(self.hlda_coherence[col]["level "+str(level)+" average"])
        index = ["total"]+["level "+str(i) for i in range(1, self.levels)]
        coherence_df = pd.DataFrame(coherence_data, index=index)
        if return_df == True:
            return coherence_df
        coherence_df.to_csv(os.path.join(self.folder_path,"hlda_coherence.csv"))
        #print("hLDA coherence scores saved to: ",self.folder_path+"/"+"hlda_coherence.csv")
    
    def save_hlda_taxonomy(self, return_df=False, use_labels=False, num_words=10):
        """ 
        Saves hlda taxonomy to file.

        Parameters
        ----------
        return_df : boolean, optional
            True to return the results df. The default is False.
        use_labels : boolean, optional
            True to use topic labels generated from tomotopy. The default is False.
        num_words : int, optional
            Number of words to display in the taxonomy. The default is 10.

        Returns
        -------
        taxonomy_df : pandas Dataframe
            Taxonomy dataframe with a column for each text column and each row a
            unique combination of topics found in the documents

        """
        
        self.__create_folder()
        taxonomy_data = {col+" Level "+str(level):[] for col in self.text_columns for level in range(1,self.levels)}
        for col in self.text_columns:
            mdl = self.hlda_models[col]
            for doc in mdl.docs: 
                topic_nums = doc.path
                for level in range(1, self.levels):
                    if use_labels == False:
                        words = ", ".join([word[0] for word in mdl.get_topic_words(topic_nums[level], top_n=num_words)])
                    else:
                        words = ", ".join(self.hlda_labels[col][topic_nums[level]])
                    taxonomy_data[col+" Level "+str(level)].append(words)
        self.taxonomy_data = taxonomy_data
        taxonomy_df = pd.DataFrame(taxonomy_data)
        taxonomy_df = taxonomy_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
        for i in range(len(taxonomy_df)):
            lesson_nums = []
            tax_row  = "\n".join([taxonomy_df.iloc[i][key] for key in taxonomy_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_data[key][j] for key in taxonomy_data])
                if doc_row == tax_row:                      
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_df["number of documents for row"] = num_lessons_per_row
        taxonomy_df = taxonomy_df.sort_values(by=[key for key in taxonomy_data])
        taxonomy_df = taxonomy_df.reset_index(drop=True)
        self.taxonomy_df = taxonomy_df
        if return_df == True:
            return taxonomy_df
        taxonomy_df.to_csv(os.path.join(self.folder_path,'hlda_taxonomy.csv'))
        #print("hLDA taxonomy saved to: ", self.folder_path+"/hlda_taxonomy.csv")
    
    def save_hlda_level_n_taxonomy(self, lev=1, return_df=False):
        """
        Saves hlda taxonomy at level n.

        Parameters
        ----------
        lev : int, optional
            the level number to save. The default is 1.
        return_df : boolean, optional
            True to return the results df. The default is False.

        Returns
        -------
        taxonomy_level_df : pandas Dataframe
            Taxonomy dataframe with a column for each text column and each row a
            unique combination of topics found in the documents

        """
        
        self.__create_folder()
        try:
            self.taxonomy_df = pd.read_csv(os.path.join(self.folder_path,'hlda_taxonomy.csv'))
        except:
            self.save_hlda_taxonomy(return_df = True)
        taxonomy_level_data = {col+" Level "+str(lev): self.taxonomy_data[col+" Level "+str(lev)] for col in self.text_columns}
        taxonomy_level_df = pd.DataFrame(taxonomy_level_data)
        taxonomy_level_df = taxonomy_level_df.drop_duplicates()
        lesson_nums_per_row = []
        num_lessons_per_row = []
        for i in range(len(taxonomy_level_df)):
            lesson_nums = []
            tax_row = "\n".join([taxonomy_level_df.iloc[i][key] for key in taxonomy_level_data])
            for j in range(len(self.doc_ids)):
                doc_row = "\n".join([taxonomy_level_data[key][j] for key in taxonomy_level_data])
                if doc_row == tax_row:                      
                    lesson_nums.append(self.doc_ids[j])
            lesson_nums_per_row.append(lesson_nums)
            num_lessons_per_row.append(len(lesson_nums))
        taxonomy_level_df["document IDs for row"] = lesson_nums_per_row
        taxonomy_level_df["number of documents for row"] = num_lessons_per_row
        taxonomy_level_df = taxonomy_level_df.sort_values(by=[key for key in taxonomy_level_data])
        taxonomy_level_df = taxonomy_level_df.reset_index(drop=True)
        if return_df == True:
            return taxonomy_level_df
        taxonomy_level_df.to_csv(os.path.join(self.folder_path,"hlda_level"+str(lev)+"_taxonomy.csv"))
        #print("hLDA level "+str(lev)+" taxonomy saved to: ", self.folder_path+"/hlda_level"+str(lev)+"_taxonomy.csv")
    
    def save_hlda_results(self):
        """
        Saves the taxonomy, level 1 taxonomy, raw topics coherence, and document topic distribution in one excel file.

        Returns
        -------
        None.

        """

        self.__create_folder()
        data = {}
        data["taxonomy"] = self.save_hlda_taxonomy(return_df=True)
        data["level 1 taxonomy"] = self.save_hlda_level_n_taxonomy(lev=1, return_df=True)
        topics_dict = self.save_hlda_topics(return_df=True)
        data.update(topics_dict)
        data["coherence"] = self.save_hlda_coherence(return_df=True)
        data["document topic distribution"] = self.save_hlda_document_topic_distribution(return_df=True)
        with pd.ExcelWriter(os.path.join(self.folder_path,"hlda_results.xlsx")) as writer2:
            for results in data:
                data[results].to_excel(writer2, sheet_name = results, index = False)
        #print("hLDA results saved to: ", self.folder_path+"/hlda_results.xlsx")
    
    def hlda_extract_models(self, file_path):
        """
        Gets hlda models from file.

        Parameters
        ----------
        file_path : string
            path to file

        Returns
        -------
        None.

        """

        self.hlda_models = {}
        self.hlda_coherence = {}
        for col in self.text_columns:
            self.hlda_models[col]=tp.HLDAModel.load(os.path.join(file_path,col+"_hlda_model_object.bin"))
            self.levels = self.hlda_models[col].depth
            self.hlda_coherence[col] = self.coherence_scores(self.hlda_models[col], "hlda")
        #print("hLDA models extracted from: ", file_path)
        preprocessed_filepath = os.path.join(file_path,"preprocessed_data")
        # if self.text_columns == ['Combined Text']:
        #     self.combine_cols = True
        #     preprocessed_filepath += "_combined_text"
        self.data.load(preprocessed_filepath+".csv", preprocessed=True, id_col=self.data.id_col, text_columns=self.data.text_columns)
        self.folder_path = file_path
        
    def hlda_display(self, col, num_words = 5, display_options={"level 1": 1, "level 2": 6}, colors='bupu', filename=''):
        """
        Saves graphviz visualization of hlda tree structure.

        Parameters
        ----------
        col : string
            column of interest
        num_words : int, optional
            number of words per node. The default is 5.
        display_options : dictionary, optional
            nested dictiary where keys are levels and values are the max number of nodes. 
            The default is {"level 1": 1, "level 2": 6}.
        colors : string, optional
            brewer colorscheme used, default is blue-purple
            see http://graphviz.org/doc/info/colors.html#brewer for options. 
            The default is 'bupu'.
        filename : string, optional
            can input a filename for where the topics are stored in order to make display 
            after hlda; must be an ouput from "save_hlda_topics()" or hlda.bin object. 
            The default is ''.

        Returns
        -------
        None.

        """
        
        try:
            from graphviz import Digraph
        except ImportError as error:
            # Output expected ImportErrors.
            print(error.__class__.__name__ + ": " + error.message)
            print("GraphViz not installed. Please see:\n https://pypi.org/project/graphviz/ \n https://www.graphviz.org/download/")
            return
        if filename != '':
            #handles saved topic inputs, bin inputs
            paths = filename.split("\\")
            self.folder_path = "\\".join([paths[i] for i in range(len(paths)-1)])
            if self.hlda_models == {}:
                self.hlda_extract_models(self.folder_path+"\\")
        try:
            df = pd.read_csv(os.path.join(self.folder_path,col+"_hlda_topics.csv"))
        except:
            try:
                df = pd.read_excel(os.path.join(self.folder_path,"hlda_results.xlsx"),sheet_name=col)
            except:
                self.save_hlda_topics()
                df = pd.read_csv(os.path.join(self.folder_path,col+"_hlda_topics.csv"))
        dot = Digraph(comment="hLDA topic network")
        color_scheme = '/'+colors+str(max(3,len(display_options)+1))+"/"
        nodes = {key:[] for key in display_options}
        for i in range(len(df)):
            if int(df.iloc[i]["topic level"]) == 0 and int(df.iloc[i]["number of documents in topic"]) > 0:
                root_words = df.iloc[i]["topic words"].split(", ")
                root_words = "\\n".join([root_words[i] for i in range(0,min(num_words,int(df.iloc[i]["number of words"])))])
                dot.node(str(df.iloc[i]["topic number"]), root_words, style="filled", fillcolor=color_scheme+str(1))
            elif int(df.iloc[i]["number of documents in topic"])>0 and str(df.iloc[i]["topic level"]) != '0':
                if (len(nodes["level "+str(df.iloc[i]["topic level"])]) <= display_options["level "+str(df.iloc[i]["topic level"])]) and not isinstance(df.iloc[i]["topic words"],float):
                    words = df.iloc[i]["topic words"].split(", ")
                    words = "\\n".join([words[i] for i in range(0,min(num_words,int(df.iloc[i]["number of words"])))])
                    topic_id = df.iloc[i]["topic number"]
                    parent_id = df.iloc[i]["parent"]
                    level = df.iloc[i]['topic level']
                    if int(level)>1 and parent_id not in nodes["level "+str(level-1)]: 
                        continue
                    else:
                        dot.node(str(topic_id), words, style="filled", fillcolor=color_scheme+str(level+1))
                        dot.edge(str(parent_id),str(topic_id))
                        nodes["level "+str(level)].append(topic_id)

        dot.attr(layout='twopi')
        dot.attr(overlap="voronoi")
        dot.render(filename = os.path.join(self.folder_path,col+"_hlda_network"), format = 'png')

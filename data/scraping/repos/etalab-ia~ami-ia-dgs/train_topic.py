"""
Auteur : Quillivic Robin, Data scientist chez StarClay, rquillivic@starclay.fr

Description: 
    Entrainement des modèles de clusterisation 

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import os
import sys
import yaml
import logging
import logging.config

with open(os.path.join(os.path.dirname(__file__), 'logging_config.yaml'), 'r') as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(log_config)

import gensim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim

from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score



path_to_regroupement = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, path_to_regroupement)
from utils import loading_function


class TopicModel:

    def __init__(self, try_name, config_dict, save_dir):
        self.data = None
        self.model = None
        self.docs = None
        self.corpus = None
        self.dictionary = None
        self.viz = None
        self.coherence = None
        self.doc_topic_mat = None

        self.try_name = try_name
        self.config_dict = config_dict
        self.save_dir = os.path.join(save_dir, 'LDA')
        os.makedirs(self.save_dir, exist_ok=True)

        log_handler = logging.FileHandler(os.path.join(self.save_dir, self.try_name + "_training.log"))
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s -  %(message)s'))
        logging.getLogger().addHandler(log_handler)
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_dictionary(self, data_df):
        """Construit l'objet dictionnaire du modèle LDA
        """
        self.data = data_df
        self.logger.info('Building dictionary')
        try:
            used_columns = self.config_dict['dictionary']['used_columns']
            print(used_columns)
            self.data['docs'] = self.data[used_columns].agg(np.sum, axis=1)
            self.logger.info('Données de construction chargées !')
        except Exception as e:
            self.logger.error(
                'Les colonnes rentrée dans le fichier de config ne correspondent pas aux données')
            raise e

        no_below = self.config_dict['dictionary']['no_below']
        no_above = self.config_dict['dictionary']['no_above']
        keep_n = self.config_dict['dictionary']['keep_n']
        self.docs = self.data['docs'].tolist()
        self.dictionary = Dictionary(self.docs)
        self.dictionary.filter_extremes(no_below=no_below,
                                        no_above=no_above,
                                        keep_n=keep_n)
        self.logger.info(f'Dictionnaire construit avec les paramètres suivant no_above = {no_below}, no_below={no_above}, kepp_n ={keep_n}')

        if self.dictionary is not None:
            self.logger.debug('saving dictionary')
            file_name = os.path.join(self.save_dir, self.try_name+'.dict')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            self.dictionary.save(file_name)
            self.logger.info(f'Dictionnaire sauvegardé dans {file_name}')

    def build_corpus(self, data_df=None):
        """Construit l'objet corpus du modèle LDA
        """
        if self.docs is None:
            if data_df is not None:
                self.build_dictionary(data_df)
            else:
                self.logger.error('No data available')
                raise RuntimeError('No Data Available')

        self.logger.info('Building corpus')
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in self.docs]
        self.corpus = doc_term_matrix
        self.logger.info('Corpus construit  utilisant une approche doc2bow! ')

        if self.corpus is not None:
            self.logger.debug('saving corpus')
            file_name = os.path.join(self.save_dir, self.try_name+'.mm')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            MmCorpus.serialize(file_name, self.corpus)
            self.logger.info(f'Corpus  sauvegardé dans {file_name}')

    def build_model(self, data_df=None):
        """Construit le modèle LDA

        Raises:
            ValueError: file not found
            ValueError: erreur lors de l'entrainement
        """
        if self.corpus is None:
            if data_df is not None:
                self.build_corpus(data_df)
            else:
                self.logger.error('No data available')
                raise RuntimeError('No Data Available')

        self.logger.info('Training model')
        try:
            name = self.config_dict['model']['name']
            if name == 'lda':
                Lda = gensim.models.LdaModel
            elif name == 'multi-lda':
                Lda = gensim.models.LdaMulticore
            elif name == 'hdp':
                Lda = gensim.models.HdpModel
            else:
                self.logger.error(f'{name} du modèle inconnu')
                raise ValueError(f'{name} est un modèle inconnu')

            nb_topics = self.config_dict['model']['num_topic']
            nb_passes = self.config_dict['model']['passes']
            # alpha = self.config_dict['model']['alpha']
            # eta = self.config_dict['model']['eta']
            self.model = Lda(self.corpus,
                             num_topics=nb_topics,
                             id2word=self.dictionary,
                             passes=nb_passes,
                             eval_every=10)
            self.logger.info(f'model construit  utilisant n_topic = {nb_topics} et {nb_passes} passes ')
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entrainment des topics: {e}")
            raise ValueError(f"Erreur lors de l'entrainment des topics:  {e}")

        if self.model is not None:
            self.logger.debug('saving model')
            file_name = os.path.join(self.save_dir, self.try_name+'.model')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            self.model.save(file_name)
            self.logger.info(f'Modèle  sauvegardé dans {file_name}')
    
    def global_save(self):
        if self.dictionary is not None:
            self.logger.debug('saving dictionary')
            file_name = os.path.join(self.save_dir, self.try_name+'.dict')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            self.dictionary.save(file_name)
            self.logger.info(f'Dictionnaire sauvegardé dans {file_name}')

        if self.corpus is not None:
            self.logger.debug('saving corpus')
            file_name = os.path.join(self.save_dir, self.try_name+'.mm')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            MmCorpus.serialize(file_name, self.corpus)
            self.logger.info(f'Corpus  sauvegardé dans {file_name}')

        if self.model is not None:
            self.logger.debug('saving model')
            file_name = os.path.join(self.save_dir, self.try_name+'.model')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            self.model.save(file_name)
            self.logger.info(f'Modèle  sauvegardé dans {file_name}')
            
        if len(self.doc_topic_mat) > 1:
                self.logger.debug('saving mat')
                file_name = os.path.join(self.save_dir, self.try_name+'.pkl')
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                self.doc_topic_mat.to_pickle(file_name)
                self.logger.info(f'doc topic matrice sauvegardée dans {file_name}')

    def build_viz(self):
        """
        Construit les élements nécessaires à l'affichage des thèmes avec la librairie pyLDAvis.
        Repose sur pyLDAvis.gensim.prepare
        """
        try:
            self.logger.info('Building visualization')
            data = pyLDAvis.gensim.prepare(
                self.model, self.corpus, self.dictionary)
            self.viz = data
        except Exception as e:
            self.logger.error(f"Erreur lors de la construction de la visualisation des topics: {e}")
            raise ValueError(f"Erreur lors de la construction de la visualisation des topics:  {e}")

        if self.viz is not None:
            self.logger.debug('saving visualization')
            file_name = os.path.join(self.save_dir, self.try_name+'.json')
            file_name_html = os.path.join(self.save_dir, self.try_name+'.html')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            pyLDAvis.save_json(self.viz, file_name)
            pyLDAvis.save_html(self.viz,file_name_html)
            self.logger.info(f'Visualisation sauvegardée dans {file_name}')

    def build_doc_topic_mat(self,save=True):
        """
        Fonction qui permet de créer la matrice thème - documents (les distribution théta dans le modèle de M. blei 2012). 
        Elle repose sur la fonction get_document_topics de gensim().

        """
        if self.data is None:
            self.logger.error('you shoud provide the document data in Mrveille format to enable the building')
            raise ValueError('you shoud provide the document data in Mrveille format to enable the building')

        self.logger.info('Building Doc-to-Topic mat')
        doc_lda = self.model.get_document_topics(self.corpus, minimum_probability=-1)
        topic_names = ['Topic'+str(i) for i in range(self.model.num_topics)]
        mat = np.array([np.array([tup[1] for tup in lst]) for lst in doc_lda])
    

        # Construction de la matrice
        df_doc_topic = pd.DataFrame(mat, columns=topic_names, index=self.data.index)
        df_doc_topic['DCO_ID'] = self.data['DCO_ID']
        df_doc_topic['text'] = self.data['text']
        df_doc_topic['text_lem'] = self.data['text_lem']
        df_doc_topic['NUMERO_DECLARATION'] = self.data['NUMERO_DECLARATION']

        self.doc_topic_mat = df_doc_topic
        # sauvegarde
        if save :
            if len(self.doc_topic_mat) > 1:
                self.logger.debug('saving mat')
                file_name = os.path.join(self.save_dir, self.try_name+'.pkl')
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                self.doc_topic_mat.to_pickle(file_name)
                self.logger.info(f'doc topic matrice sauvegardée dans {file_name}')
        

    def load(self, filename):
        """
        Charge une configuration de TopicModel, c'est à dire le dictionnaire, le corpus, le model, la visualisation  et la matrice associés
        Args:
            filename (str): le nom de la config qui sert de nom de fichier

        Raises:
            ValueError: le chemin spécifié n'est pas le bon
        """
        self.logger.info(f'Loading from {os.path.join(self.save_dir, filename)}')
        # Dict
        path_dict = os.path.join(self.save_dir, filename+'.dict')
        self.dictionary = loading_function(path_dict, Dictionary.load,
                                           self.dictionary, self.logger)
        # Corpus
        path_corpus = os.path.join(self.save_dir, filename+'.mm')
        self.corpus = loading_function(
            path_corpus, MmCorpus, self.corpus, self.logger)
        # Model
        path_model = os.path.join(self.save_dir, filename+'.model')
        self.model = loading_function(path_model, gensim.models.LdaModel.load,
                                      self.model, self.logger)
        # viz
        with open(os.path.join(self.save_dir, filename+'.json'), 'r') as json_file:
            self.viz = loading_function(
                json_file, json.load, self.viz, self.logger)

        # doc topic matrice
        path_mat = os.path.join(self.save_dir, filename+'.pkl')
        self.doc_topic_mat = loading_function(path_mat, pd.read_pickle,
                                              self.doc_topic_mat, self.logger)

    def set_data(self, data_as_df):
        self.data = data_as_df
        used_columns = self.config_dict['dictionary']['used_columns']
        self.data['docs'] = self.data[used_columns].agg(np.sum, axis=1)
        self.docs = self.data['docs'].tolist()


    def get_coherence_score(self, save=False):
        """ Calcul du score de cohérence d'un modèle de topic avec le paramètre u_mass, base sur ce pipelie: 
        http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf

        Returns:
            coherence (float): Score de cohérence du modèle
        """
        coherence_model_lda = CoherenceModel(
            model=self.model, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
        self.coherence = coherence_model_lda.get_coherence()
        if save:
            res = {'config': self.config_dict,
                   'coherence_score': self.coherence}
            with open(os.path.join(self.save_dir, self.try_name+'_results.json'), 'w') as f:
                json.dump(res, f)
        return self.coherence

    def build_topic_mat(self, k=50):
        """Construit ma matrice des distances inter-topic.
        Utilise la distance de Jaccard et les k premiers mots du topic.
        Basé sur mdiff de gensim

        Args:
            k (int, optional): Nombre de mots à utiliser pour représenter un topic. Defaults to 50.
        """
        mdiff, annotation = self.model.diff(
            self.model, distance='jaccard', num_words=k)
        self.mdiff = mdiff

    def predict(self, new_docs_as_df):
        """inférer le poids des topics dans chaque document

        Args:
            doc (pd.dataframe): dataframe au format Mrveill comportant un document

        Returns:
            doc_lda (np.array) : distrubution des thèmes sur le documents données
        """

        used_columns = self.config_dict['dictionary']['used_columns']
        print(used_columns)
        if type(new_docs_as_df)==pd.core.series.Series:
            new_docs  = new_docs_as_df[used_columns].agg(np.sum)
            new_corpus =  [self.dictionary.doc2bow(new_docs)]
            index = [0]
        else :
            new_docs = new_docs_as_df[used_columns].agg(np.sum,axis=1).tolist()
            new_corpus = [self.dictionary.doc2bow(doc) for doc in new_docs]
            index = new_docs_as_df.index
        
        doc_lda = self.model.get_document_topics( new_corpus, minimum_probability=-1)
        topic_names = ['Topic'+str(i) for i in range(self.model.num_topics)]
        mat = np.array([np.array([tup[1] for tup in lst]) for lst in doc_lda])

        # Construction de la matrice
        df_doc_topic = pd.DataFrame(
            mat, columns=topic_names, index=index)
        df_doc_topic['DCO_ID'] = new_docs_as_df['DCO_ID']
        df_doc_topic['text'] = new_docs_as_df['text']
        df_doc_topic['text_lem'] = new_docs_as_df['text_lem']
        df_doc_topic['NUMERO_DECLARATION'] = new_docs_as_df['NUMERO_DECLARATION']

        return df_doc_topic

    def update(self, new_docs_as_df):
        """inférer le poids des topics dans chaque document

        Args:
            doc (pd.dataframe): dataframe au format Mrveill comportant un document

        Returns:
            doc_lda (np.array) : distrubution des thèmes sur le documents données
        """

        used_columns = self.config_dict['dictionary']['used_columns']
        print(used_columns)

        if self.data is None:
            self.logger.error('You need to set the data to the old training data first')
            raise RuntimeError('You need to set the data to the old training data first')
        self.set_data(pd.concat((self.data, new_docs_as_df), ignore_index=True))

        # update dictionary
        new_docs = new_docs_as_df[used_columns].agg(np.sum,axis=1).tolist()
        # self.dictionary.add_documents(new_docs)

        # update corpus
        corpus = [doc for doc in self.corpus]
        new_corpus = [self.dictionary.doc2bow(doc) for doc in new_docs]
        self.corpus = corpus + new_corpus
        
        doc_lda = self.model.update(new_corpus)
        self.logger.info('Modèle mis à jour avec les nouvelles données')

    
    def evaluate(self,save=False):
        """Permet d'évaluer la capacité d'un topic modèl a classifier les dco à l'aide d'un svm

        Args:
            save (bool, optional): sauvegarde t'on le score ?. Defaults to False.

        Returns:
            score (float): balanced accuracy sur la classification des dcos
        """
        
        
        classifier = LinearSVC(class_weight='balanced')
        n = self.model.num_topics
        X = self.doc_topic_mat.iloc[:, :n-1]
        y = self.doc_topic_mat['DCO_ID'].fillna('0').astype(int)

        train_index, test_index = next(GroupShuffleSplit(random_state=1029).split(X, groups=self.doc_topic_mat['text']))
        X_train, X_test = X.iloc[train_index,:].values, X.iloc[test_index,:].values
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        self.score = balanced_accuracy_score(y_test, y_pred)

        if save : 
            res = {'config': self.config_dict,
                   'coherence_score': self.score}
            with open(os.path.join(self.save_dir, self.try_name+'_evaluation.json'), 'w') as f:
                json.dump(res, f)


        return self.score


if __name__ == "__main__":
    import shutil

    current_dir = os.path.dirname(__file__)
    config_file = 'training_config.yaml'

    if len(sys.argv) == 2:
        config_file = sys.argv[1]

    if not os.path.isabs(config_file):
        config_file = os.path.abspath(os.path.join(current_dir, config_file))

    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    try_name = config['config_name']
    save_dir = os.path.join(config['path_to_save'], try_name)
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(os.path.join(current_dir, save_dir))

    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_file, save_dir)

    topic_model = TopicModel(try_name, config['topic'], save_dir=save_dir)

    filename = os.path.join(config['data']['path'], config['data']['filename'])
    if not os.path.isabs(filename):
        filename = os.path.abspath(os.path.join(current_dir, filename))
    n_ligne = None
    try:
        df = pd.read_pickle(filename)
        if n_ligne is not None:
            df = df.iloc[:n_ligne, :]
        topic_model.logger.info('Chargement des données ! Ok !')
    except Exception as e:
        topic_model.logger.error(f'Error loading {filename}: {e}')
        raise ValueError(e)

    topic_model.build_dictionary(df)
    topic_model.build_corpus()
    topic_model.build_model()
    topic_model.build_viz()
    TopicModel.build_doc_topic_mat()
    topic_model.logger.info(f'Coherence score : {topic_model.get_coherence_score(save=True)}')

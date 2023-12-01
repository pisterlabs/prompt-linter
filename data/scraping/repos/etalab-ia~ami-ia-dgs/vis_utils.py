"""
Auteur : Quillivic Robin, Data Scientist, rquillivic@starclay.fr 
"""

import pandas as pd
import gensim
import numpy as np
import os
import pyLDAvis
import pyLDAvis.gensim


from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import joblib
from sklearn import metrics


from sklearn.cluster import FeatureAgglomeration,DBSCAN, OPTICS,KMeans
from sklearn.metrics import pairwise_distances


class ClusteurModel():
    def __init__(self):
        self.topicmodel = TopicModel()
        self.n_clusteur = 150
        self.model = None
        self.dist = None
        
        
    def train(self):
        n = self.topicmodel.model.num_topics
        self.X = self.topicmodel.doc_topic_mat.iloc[:,0:n-1].values
        agglo = KMeans(self.n_clusteur)
        agglo.fit(self.X)
        self.model = agglo
    
    def save(self,path_to_file):
        joblib.dump(self.model,path_to_file)

    def load(self,path_to_file,filename):
        n = self.topicmodel.model.num_topics
        self.model = joblib.load(os.path.join(path_to_file,filename))
        self.X = self.topicmodel.doc_topic_mat.iloc[:,0:n-1].values
        
    def compute_score(self): 
        self.silhouette_score = metrics.silhouette_score(self.X, self.model.labels_, metric='euclidean')
        self.calinski_harabasz_score = metrics.calinski_harabasz_score(self.X, self.model.labels_)
        self.davies_bouldin_score = metrics.davies_bouldin_score(self.X, self.model.labels_)
    
    def build_dist_mat(self):
        self.dist = pairwise_distances(self.model.cluster_centers_, metric = 'cosine')
        return self.dist
    
    def plot_dist_mat(self,title ="Distance cosinus inter clusteur"): 
        fig, ax = plt.subplots(figsize=(18, 14))
        data = ax.imshow(self.dist, cmap='RdBu_r', origin='lower')
        plt.title(title)
        plt.colorbar(data)
    
    def get_furthest(self):
        maxi = self.dist.max()
        c = np.where(self.dist == maxi)
        return c
    
    def get_closest(self):
        mini = np.min(self.dist[np.nonzero(self.dist)])
        c= np.where(self.dist == mini)
        return c
                      
        
        
        

import yaml
from spacy.lang.fr.stop_words import STOP_WORDS 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotly import graph_objs as pgo

class AnalysedClusteur: 
    def __init__(self):
        self.clusteur = pd.DataFrame
        self.seuil = 0.1
        self.top_dcos = None
        self.top_topics = None
        self.wc = None
        self.topicmodel = TopicModel()
        self.id_to_dco = None
    
    def load(self,path_to_conf_file,filename):
        
        with open(os.path.join(path_to_conf_file,'config.yaml'), 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        
        self.id_to_dco = pd.read_csv(os.path.abspath(os.path.join(config['data']['id_to_dco']['file_path'])
            ), delimiter=';', encoding='ISO-8859-1')
        self.topicmodel.load(config['topic']['saved_path'],filename)
         
        
    def get_significant_topic(self,x):
        return np.where(x>self.seuil)

    def get_top_topics(self,k=5):
        n = self.topicmodel.model.num_topics
        T = self.clusteur.iloc[:,0:n-1].apply(lambda x: self.get_significant_topic(x), axis=1)
        topic = np.concatenate(T.values,axis=1)
        c = pd.DataFrame.from_dict(Counter(topic[0]),orient='index',columns=['count'])
        df_top_topic = c.sort_values(by='count',ascending=False).iloc[:k]
        most_frequent_topic = df_top_topic.index
        weight = df_top_topic.values/df_top_topic.values.sum()
        w= [elt[0] for elt in weight]
        topic = []
        for elt in most_frequent_topic :
            word = [t[0] for t in self.topicmodel.model.show_topic(elt)[:2]]
            word_str = ' '.join(word)
            title = 'Topic: '+str(elt)+ '(' +word_str +'...)'
            topic.append(title)
        df = pd.DataFrame()
        df['top_topics'] = topic
        df['%'] = [100*x for x in w]
        self.top_topics = df
        return  df

    def get_top_dcos(self):
        c = self.clusteur.groupby('DCO_ID').count()['Topic0']
        df =c.sort_values(ascending=False).iloc[:5]
        most_frequent_dco = df.index
        weight = df.values/df.values.sum()
        #w= [elt[0] for elt in weight]
        dcos = []
        for elt in most_frequent_dco :
            dco = self.id_to_dco[self.id_to_dco['DCO_ID'] == int(elt)]['LIBELLE'].iloc[0]
            dcos.append(dco)

        df_top_dcos = pd.DataFrame()
        df_top_dcos['top_dcos'] = dcos
        df_top_dcos['%'] = [100*x for x in weight]
        self.top_dcos = df_top_dcos
        return df_top_dcos
    
    def prepare_cluster_plot(self,clustermodel):
        df_doc_topic = self.topicmodel.doc_topic_mat
        df_doc_topic['cluster'] = clustermodel.model.labels_
        group = df_doc_topic.groupby('cluster')
        
        X = clustermodel.model.cluster_centers_
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)
        
        df_cluster = pd.DataFrame()
        df_cluster['cluster'] = group['text'].count().index
        df_cluster['weight'] = group['text'].count().tolist()
        df_cluster['X'] = X_reduced[:,0]
        df_cluster['Y'] = X_reduced[:,1]
        
        return df_cluster
    
    def plot_cluster(self,df_cluster): 
        trace = pgo.Scatter(x= df_cluster['X'],
                    y=df_cluster['Y'],
                    text=df_cluster.index,
                    mode='markers',
                    #
                    marker=pgo.scatter.Marker(size=df_cluster['weight'],
                                      sizemode='diameter',
                                      sizeref=df_cluster['weight'].max()/50,
                                      opacity=0.5)
)

        
        layout5 = pgo.Layout(title='Distribution des Clusters (PCA n=2)',
                     xaxis=pgo.layout.XAxis(showgrid=True,
                                     zeroline=True,
                                     showticklabels=True),
                     yaxis=pgo.layout.YAxis(showgrid=True,
                                     zeroline=True,
                                     showticklabels=True),
                     hovermode='closest'
                    )

        fig5 = pgo.Figure(layout=layout5)
        fig5.add_trace(trace)
        fig5.layout.update(height = 500,width =1000)
        fig5.show()
        
    
    def build_wc(self): 
        text = np.sum(self.clusteur['text'].values)
        wc = WordCloud(background_color="white", stopwords=STOP_WORDS,width=1000, height=500, max_words = 30).generate(text)
        self.wc = wc
        
    def plot_wc(self):
        # lower max_font_size
        plt.figure(figsize=(10,20))
        plt.imshow(self.wc, interpolation="bilinear")
        plt.axis("off")    
        plt.show()
        
    def convert_wc_to_plotly(self): 
        # Transforme en plotly image    
        word_list=[]
        freq_list=[]
        fontsize_list=[]
        position_list=[]
        orientation_list=[]
        color_list=[]

        for (word, freq), fontsize, position, orientation, color in self.wc.layout_:
            word_list.append(word)
            freq_list.append(freq)
            fontsize_list.append(fontsize)
            position_list.append(position)
            orientation_list.append(orientation)
            color_list.append(color)
        
        # Prendre les positions
        x=[]
        y=[]
        for i in position_list:
            x.append(i[0])
            y.append(i[1])
            
        # Obtenir les fréquences relatives
        new_freq_list = []
        for i in freq_list:
            new_freq_list.append(i*100)
        new_freq_list
        
        return x,y, new_freq_list, color_list,word_list,freq_list


from _plotly_future_ import v4_subplots
from plotly.subplots import make_subplots
#from plotly import express
from plotly import graph_objs as go
from pprint import pprint


     
def ClusterAnalysis(cluster: pd.DataFrame) :
        analyse = AnalysedClusteur()
        analyse.clusteur  = cluster
        analyse.load('/home/robin/Nextcloud/strar_clay/GitLab/signalement-ia/regroupement/exploration/', 'rake_tri')
        df_topics = analyse.get_top_topics()
        df_dcos = analyse.get_top_dcos()
        analyse.build_wc()
        wc = analyse.wc 
        #analyse.plot_wc()
        #x,y, new_freq_list, color_list,word_list,freq_list = analyse.convert_wc_to_plotly()
        return(wc,df_dcos,df_topics)



def plotClusterAnalysis(cluster: pd.DataFrame): 
    wc,df_dcos,df_topics = ClusterAnalysis(cluster)
    fig = plt.figure(figsize =(7,15))
    
    ax = fig.add_subplot(411)
    df_topics.sort_values('%').plot(x='top_topics', y = '%',kind='barh', ax=ax,title='Distributuion des Topics')

    ax2 = fig.add_subplot(412)
    df_dcos.sort_values('%').plot(x = 'top_dcos',y='%',kind='barh',ax=ax2, title = 'Distribution des DCO',color = 'orange')
    
    plt.subplot(413)
    plt.imshow(wc, interpolation="bilinear")
    plt.title('Nuage des mots les plus présents dans ce cluster')
    plt.axis("off")
    
    plt.subplot(414)
    plt.text(0,0.3,"**Document type :** \n"+ str(),fontsize=15,wrap=True)
    plt.title('Document le plus représentatis du cluster sélectionnés')
    plt.axis('off')

    plt.suptitle("Analyse d'un cluster")

    plt.show()

def CompareClusterAnalysis(cluster1,cluster2): 
    wc1,df_dcos1,df_topics1 = ClusterAnalysis(cluster1)
    wc2,df_dcos2,df_topics2 = ClusterAnalysis(cluster2)
    fig = plt.figure(figsize =(20,15))
    
    ax = fig.add_subplot(3,2,1)
    df_topics1.sort_values('%').plot(x='top_topics', y = '%',kind='barh', ax=ax,title='Distributuion des Topics')
    
    ax21 = fig.add_subplot(3,2,2)
    df_topics2.sort_values('%').plot(x='top_topics', y = '%',kind='barh', ax=ax21,title='Distributuion des Topics')
    
    ax2 = fig.add_subplot(3,2,3)
    df_dcos1.sort_values('%').plot(x = 'top_dcos',y='%',kind='barh',ax=ax2, title = 'Distribution des DCO',color = 'orange')
    
    ax22 = fig.add_subplot(3,2,4)
    df_dcos2.sort_values('%').plot(x = 'top_dcos',y='%',kind='barh',ax=ax22, title = 'Distribution des DCO',color = 'orange')

    plt.subplot(3,2,5)
    plt.imshow(wc1, interpolation="bilinear")
    plt.title('Nuage des mots les plus présents dans ce cluster')
    plt.axis("off")
    
    plt.subplot(3,2,6)
    plt.imshow(wc2, interpolation="bilinear")
    plt.title('Nuage des mots les plus présents dans ce cluster')
    plt.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Comparaison de deux clusters")

def plotClusterAnalysisPlotly(cluster : pd.DataFrame):
    wc,top_topics,topic_weight,top_dcos,dcos_weight,x,y, new_freq_list, color_list,word_list,freq_list =  ClusterAnalysis(cluster)
    fig = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [{"colspan": 2}, None]],
    subplot_titles=("Distribution des Topics","Distribution des DCO", "Nuage de mot "))
    
    fig.add_trace(go.Bar(x = topic_weight ,y = top_topics ,orientation='h'), row=1, col=1)
    
    fig.add_trace(go.Bar(x=dcos_weight, y=top_dcos,orientation ='h'),
                 row=2, col=1)

    fig.add_trace(plt.imshow(wc, interpolation="bilinear"),row=3, col=1)

    fig.layout.update(height = 1000,width =500, showlegend=False, title_text="Analyse d'un cluster")
    #go.Scatter(x=x,y=y, textfont = dict(size=new_freq_list, color=color_list),
                       #hoverinfo='text',
                       #hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)], mode='text',  text=word_list)
    fig.update_yaxes(automargin=True)
    fig.show()
    


from gensim.corpora import Dictionary,MmCorpus
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import json


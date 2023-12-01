"""
Auteur :  Quillivic Robin, Data Scientist chez StarClay, rquillivic@starclay.fr

Description :  
Génère un fichier excel pour l'analyse des algorithmes du Livralbe 3
Le fichier Execel généré comporte 3 feuille :
- la feuille 'thèmes', qui contient 10 des topics les moins bons, 10 des clusters les  plus cohérents et 10 moyennement cohérent
- la feuille 'clusters', contient les indices de 50 clusters sélectionnés au hasard
- la feuille ' Documents, contient les identifiants de 20 documents selectionés de manière aléatoire

Pour générer le fichier Excel, il faut préciser dans le fichier  analyse_config.yaml, le nom de l'analyse pour laquel il faut géner le fichie excel. 
Puis exécuter python3 generate_xlsx.py

"""

import pandas as pd
import gensim
import pyLDAvis
import pyLDAvis.gensim
import os
import sklearn as sk 
import json
import numpy as np
import yaml 

import sys
path_to_regroupement = os.path.dirname(os.path.dirname(__file__))

sys.path.insert(1,os.path.join(path_to_regroupement, 'training/' ))
import train_topic, train_cluster



with open(os.path.join(path_to_regroupement,'analyse','analyse_config.yaml'), 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)
    
with open(os.path.join(path_to_regroupement, 'config.yaml'), 'r') as stream:
    globale_config = yaml.load(stream, Loader=yaml.FullLoader)
    
config_path = os.path.join(os.path.join(path_to_regroupement, 'training'),'training_config.yaml')
with open(config_path, 'r') as stream:
    config_training = yaml.load(stream, Loader=yaml.FullLoader)

path_mrv = globale_config['data']['mrv']['path']
data_mrv = pd.read_csv(path_mrv)


SAVE_PATH = config_data['analyse']['path']
path = config_data['analyse']['path']
analyse_name = config_data['analyse']['name']
cluster_name = analyse_name

# Loading
topic_model = train_topic.TopicModel(analyse_name, config_training['topic'], 
                                        save_dir=os.path.join(SAVE_PATH, analyse_name))
topic_model.load(analyse_name)

cluster_model = train_cluster.ClusterModel(cluster_name, config_training['cluster'],
                                        save_dir=os.path.join(SAVE_PATH, cluster_name))
cluster_model.topicmodel = topic_model
cluster_model.load(cluster_name)

n_topics = 30
n_cluster = 50
n_doc = 20

def build_topic_list(topicmodel,n_topics) :
    """Sélectionne les topics à analyser: n_topics/3 très cohérent, 10 pas cohérent et n_topics/3 moyennement cohérent

    Args:
        topicmodel (TopicModel): [description]
        n_topics (int): Nombre de topic à analyser

    Returns:
        topic_list (list(int)): Liste des topic à analyser
    """
    k = int(n_topics/2)
    from gensim.models import CoherenceModel
    coherence_model_lda = CoherenceModel(
    model=topicmodel.model, corpus=topicmodel.corpus, dictionary=topicmodel.dictionary, coherence='u_mass')
    df = pd.DataFrame(index = np.arange(0,topicmodel.model.num_topics))
    c = coherence_model_lda.get_coherence_per_topic()
    df['c'] = c 
    top_10 = [elt+1 for elt in df.sort_values('c',ascending=False).index[:k].tolist()]
    w_10 = [elt+1 for elt in df.sort_values('c',ascending=True).index[:k].tolist()]
    med_10 = [elt+1 for elt in df[(df['c'] >df['c'].quantile(0.45)) & (df['c'] <df['c'].quantile(0.55))].index.tolist()]
    
    return (top_10 +w_10+med_10)

num_clusters = len(set(cluster_model.model.labels_))
cluster_list = np.random.randint(0,num_clusters,n_cluster)
topic_list = build_topic_list(topic_model,n_topics)

document_list = np.random.randint(0,len(data_mrv),n_doc)


# Topic sheet
df_topic = pd.DataFrame()
df_topic['Numero'] = ['Topic '+str(x) for x in topic_list]
df = pd.DataFrame.from_dict(topic_model.viz['token.table'])
df_topic['mots'] = [' '.join(df.groupby('Topic').get_group(x).sort_values(by='Freq', ascending=False)['Term'].iloc[:5].tolist()) for x in topic_list]
#df_topic['mots'] = [' '.join([elt[0] for elt in topic_model.model.show_topic(x-2)[:5]]) for x in topic_list]
df_topic['score'] = np.nan
df_topic['titre'] = np.nan

#cluster sheet
df_cluster = pd.DataFrame()
df_cluster['Numero'] = ['Cluster '+str(x) for x in cluster_list]
df_cluster['score'] = np.nan
df_cluster['titre'] = np.nan 

#Document 
df_doc = pd.DataFrame()
df_doc[['NUMERO_DECLARATION','DCO']] = data_mrv.iloc[document_list][['NUMERO_DECLARATION','DCO']]
df_doc['score'] = np.nan 


writer = pd.ExcelWriter('analyse.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_topic.to_excel(writer, sheet_name='thèmes')
df_cluster.to_excel(writer, sheet_name='clusters')
df_doc.to_excel(writer, sheet_name='documents')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
'''
Created on Aug 9, 2018

@author: yingc
'''
from gensim import corpora, models, similarities
from pprint import pprint
import matplotlib.pyplot as plt

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import numpy as np
from treelib import  Tree

from gensim.models.coherencemodel import CoherenceModel


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from numpy.random.mtrand import RandomState
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
import queue

import os
import networkx as nx
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score

file_dir = 'E:\\Reddit\\data'

def file_reader(file_ID):
    filename = file_ID + '.txt'
    filepath_name = os.path.join(file_dir+'\\labeled data\\', filename)
    
    ndtype = 'i, i, i, S100, S2000' 
    names = 'Idx, Pidx, label, Topic, Content'
    ps = np.genfromtxt(filepath_name, dtype=ndtype, names=names, delimiter='\t', encoding='utf_8_sig')

    edges = zip(ps['Idx'], ps['Pidx'])
   
    label = ps['label']
    order =  ps['Idx']
    content = ps['Content']

    d=[]
    
    for item in edges:
        d.append(item)
        g = nx.Graph(d)
        
        #pos = nx.shell_layout(g)
    #nx.draw(g)
        nx.draw_networkx(g)
        
        #plt.pause(0.5)
        plt.clf()
    #plt.show()

    return edges, label, content

def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

def get_degree_height(edges, label):
    tree = Tree()
    tree.create_node(edges[0][0], edges[0][0],data=label[0])  # root node
    
    for i in range(len(edges[1:])):
        tree.create_node(tag=edges[1:][i][0], identifier = edges[1:][i][0], parent=edges[1:][i][1], data = label[i+1])
    
    tree.show()
    #tree_height = max([len(item) for item in tree.paths_to_leaves()])-1
    print edges
    
    node_heights = []
    node_degrees = []

    for i in range(len(edges)):
        node_height = max([len(item) for item in tree.subtree(i).paths_to_leaves()])
        node_heights.append(node_height)
        node_degree  = len(tree.subtree(i).nodes)
        node_degrees.append(node_degree)

    '''
    for edge in edges:
        print tree.get_node(edge[0])
        print tree.get_node(edge[0]).fpointer
        print tree.get_node(edge[0]).bpointer
        print tree.level(edge[0])
        node_heights.append(tree_height- tree.level(edge[0]))
        node_degrees.append(len(tree.get_node(edge[0]).fpointer))
    '''
    node_degrees = Normalization(np.array(node_degrees))
    node_heights = Normalization(np.array(node_heights))
    
    X = zip(node_degrees,node_heights)
    #X = zip(node_degrees,order)

    print 0.66*np.array(node_heights)+0.34*np.array(node_degrees)
    
    return X

if __name__ == '__main__':
    
    edges1, label1, documents1 = file_reader("867njq_new")
    edges2, label2, documents2 = file_reader("8sk1ue")
    
    tree_degree_height1 = get_degree_height(edges1, label1)
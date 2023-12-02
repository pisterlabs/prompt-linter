# -*- coding:utf-8 -*-
'''
Created on Apr 16, 2018

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

import os
import networkx as nx
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score
from datetime import datetime
from collections import Counter
from gensim.test.test_sklearn_api import texts


#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file_dir = 'E:\\Reddit\\data'


simple0= ["lBJ Lakers",
            "Warriors Championship"]

simple1= ["lBJ LBJ Lakers Lakers",
            "Warriors Championship"]

simple2= ["lBJ LBJ Lakers Lakers",
            "Warriors  Warriors Championship  Championship"]

simple3= ["lBJ LBJ Lakers Lakers LBJ LBJ Lakers Lakers",
            "Warriors  Warriors Championship  Championship"]

simple4 = ["Texas serial bomber made video confession before blowing himself up",
              "What are the chances we ever see the video?",
              "About the same as the chances of the Browns winning the Super Bowl.",
              "every morning.",
              "I have to applaud your regularity",
              "Pshh I'm taking the browns to the super bowl as we speak",
              "Consistency is the key.",
              "Seriously. Well done.",
              "Zero, videos like this are locked down and used for training purposes. There are a host of confessions and tapes of crimes the public will never see and some have caused agents in training to kill themselves because they are so vile.",
              "here I am thinking 'just transcripts? How bad can it be' Bad, guys. Very bad."
              
            ]

simple6= [  "Warriors got the Championship",
            "Yeah, they deserve it!",
            "lBJ went to Lakers",
            "shit, that's so bad. I cannot believe it.",
            "Oh my gosh, I will not watch Cavs's game"]

edges6 = {0:0,
          1:0,
          2:2,
          3:2,
          4:2
         }

edges66 = [(0,0),(1,0),(2,0),(3,2),(4,2)]
lable66= [1,0,1,0,0]
         
lable666= [2,1,0,2,0]
#edges6_pairswitch = dict((value,key) for key,value in edges6.iteritems())

edges= edges66
label = lable666
#documents = simple6


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

#print 'got', len(documents), 'documents'    # got 9 documents
#pprint(documents)

def prefix_time():
    prefix_time = datetime.now().strftime('%Y%m%d%H%M%S')
    return prefix_time

def file_reader(file_ID):
    filename = file_ID + '.txt'
    filepath_name = os.path.join(file_dir+'\\labeled data\\', filename)
    
    ndtype = 'i, i, i, i, S2000' 
    names = 'Idx, Pidx, label, Topic, Content'
    ps = np.genfromtxt(filepath_name, dtype=ndtype, names=names, delimiter='\t', encoding='utf_8_sig')

    edges = zip(ps['Idx'], ps['Pidx'])
   
    label = ps['label']
    order =  ps['Idx']
    content = ps['Content']
    topic = ps['Topic']
    
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

    return edges, label, content, topic

def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
    #return [(float(i))/float(max(x)) for i in x]
    #return [0.1+ (float(i)-min(x))/float(max(x)-min(x))*(0.9-0.1) for i in x]

def get_degree_height(edges, label):
    tree = Tree()
    tree.create_node(edges[0][0], edges[0][0], data = label[0])  # root node
    parents = []
    parents.append(0)
    for i in range(len(edges[1:])):
        tree.create_node(tag=edges[1:][i][0], identifier = edges[1:][i][0], parent=edges[1:][i][1], data = label[i+1])
        if tree.parent(i):
            parents.append(tree.parent(i).identifier)
            
    parents.append(tree.parent(len(edges[1:])).identifier)
    
    tree.show()
   
    #tree_height = max([len(item) for item in tree.paths_to_leaves()])-1

    node_heights = []
    node_degrees = []
   
    node_popularity = []
    
    #difference
    H = tree.depth()+1
    
    #base
    a = 0.8
    
    #Gravity factor
    G = 1
    for i in range(len(edges)):
        
        node_height = max([len(item) for item in tree.subtree(i).paths_to_leaves()])
        node_heights.append(node_height)
        
        node_degree  = len(tree.subtree(i).nodes)
        node_degrees.append(node_degree)
        
        subtrees = tree.subtree(i).paths_to_leaves()
        
        h = node_height
        node_set = set()
        weight = 0
        for subt in subtrees:
            for index, node in enumerate(subt): 
                #w = (h-index)/float(h)
                
                #Arithmetic Progression
                #w = 1 - float(index)/H
                
                #Geometric Progression
                #w = 1 * float(a)**index
               
                #Harmonic Progression
                w = float(1)/(index+1)**G
                
                if node not in node_set:
                    weight += w
                    node_set.add(node)
        node_popularity.append(weight)
    print node_popularity  
    '''
    for edge in edges:
        node_heights.append(tree_height- tree.level(edge[0]))
        node_degrees.append(len(tree.get_node(edge[0]).fpointer))
    '''
    #print node_degrees
    #print node_heights
    #print node_popularity
    node_degrees = Normalization(np.array(node_degrees))
    node_heights = Normalization(np.array(node_heights))

    #X = zip(node_degrees,node_heights)
    X =  [[i] for i in node_degrees]
    #X = node_degrees
    #X = zip(node_degrees,order)

    #extension = 0.66*np.array(node_heights)+0.34*np.array(node_degrees)
    #print node_degrees
    #print node_heights
    #extension = 0.5*np.array(node_heights)+0.5*np.array(node_degrees)
   
    return X, parents

def train_with_degree_height(X,y):
    #clf = LinearSVC(random_state= 0)
    
    X = X
    y = y

    '''
    x1 = np.random.randn(100)
    x2 = 4*np.random.randn(100)
    x3 = 0.5*np.random.randn(100)
    y = (3 + x1 + x2 + x3 + 0.2*np.random.randn()) > 0
    X = np.column_stack([x1, x2, x3])
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    #print X_train
    #print OneVsRestClassifier(clf).fit(X_train, y_train).predict(X_test)
   
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    print "LogisticRegression:"
    
    prob_y_2 = model.predict_proba(X_test)
    # Keep only the positive class
    prob_y_2 = [p[1] for p in prob_y_2]
    print "roc_auc_score:", ( roc_auc_score(y_test, prob_y_2) )
    
    print "accuracy:" + str(accuracy_score(y_test, y_predict))
    # The estimated coefficients will all be around 1:
    print(model.coef_)
  
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print "DecisionTreeClassifier:"
    
    prob_y_2 = model.predict_proba(X_test)
    # Keep only the positive class
    prob_y_2 = [p[1] for p in prob_y_2]
    print "roc_auc_score:",  ( roc_auc_score(y_test, prob_y_2) )
    
    print "accuracy:" + str(accuracy_score(y_test, y_predict))
    print(model.feature_importances_)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print "RandomForestClassifier:"
    
    prob_y_2 = model.predict_proba(X_test)
    # Keep only the positive class
    prob_y_2 = [p[1] for p in prob_y_2]
    print "roc_auc_score:",  ( roc_auc_score(y_test, prob_y_2) )
    
    print "accuracy:" + str(accuracy_score(y_test, y_predict))
    print(model.feature_importances_)
    
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print "ExtraTreesClassifier:"
    
    prob_y_2 = model.predict_proba(X_test)
    # Keep only the positive class
    prob_y_2 = [p[1] for p in prob_y_2]
    print "roc_auc_score:",  ( roc_auc_score(y_test, prob_y_2) )
    
    print "accuracy:" + str(accuracy_score(y_test, y_predict))
    print(model.feature_importances_)

    model = LinearSVC(random_state= 0, class_weight="balanced")
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print "LinearSVC:"
    
    print "accuracy:" + str(accuracy_score(y_test, y_predict))
    print(model.coef_)


   

    '''
    clf.fit(X_train, y_train)  
    print clf.predict(X_test)
    print y_test
    '''
'''
def cross_training(X1,y1,X2,y2):
    clf = LinearSVC(random_state= 0)
    X = np.array(X1)
    y = np.array(y1)
    #X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.5)
    #print X_train
    #print OneVsRestClassifier(clf).fit(X_train, y_train).predict(X_test)
    y_predict = clf.fit(X1, y1).predict(X2)
    print clf.coef_
    print y2
    print y_predict
    
    print "accuracy:" + str(accuracy_score(y2, y_predict))
'''
def cross_training(X1,y1,X2,y2):
    
    X = X1
    y = y1
    #X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.5)
    #print X_train
    #print OneVsRestClassifier(clf).fit(X_train, y_train).predict(X_test)
   
    print "------------------------------------------------"
    model = LogisticRegression()
    model.fit(X1, y1)
    y_predict = model.fit(X1, y1).predict(X2)
    print "LogisticRegression:"
    print "accuracy:" + str(accuracy_score(y2, y_predict))
    # The estimated coefficients will all be around 1:
    print(model.coef_)
  
    
    model = DecisionTreeClassifier()
    model.fit(X1, y1)
    # display the relative importance of each attribute
    y_predict = model.fit(X1, y1).predict(X2)
    print "DecisionTreeClassifier:"
    print "accuracy:" + str(accuracy_score(y2, y_predict))
    print(model.feature_importances_)
    
    model = RandomForestClassifier()
    model.fit(X1, y1)
    # display the relative importance of each attribute
    y_predict = model.fit(X1, y1).predict(X2)
    print "RandomForestClassifier:"
    print "accuracy:" + str(accuracy_score(y2, y_predict))
    print(model.feature_importances_)
    
    model = ExtraTreesClassifier()
    model.fit(X1, y1)
    # display the relative importance of each attribute
    y_predict = model.fit(X1, y1).predict(X2)
    print "ExtraTreesClassifier:"
    print "accuracy:" + str(accuracy_score(y2, y_predict))
    print(model.feature_importances_)

    model = LinearSVC(random_state= 0)
    model.fit(X1, y1)
    y_predict = model.fit(X1, y1).predict(X2)
    print "LinearSVC:"
    print "accuracy:" + str(accuracy_score(y2, y_predict))
    print(model.coef_)

    
class MyTexts(object):
    """Construct generator to avoid loading all docs
    
    """
    def __init__(self, documents):
        #stop word list
        #self.stoplist = set('for a of the and to in'.split())
        self.documents = documents
        print 'got', len(documents), 'documents'  

    def __iter__(self):
        for doc in self.documents:
            #remove stop words from docs
            stop_free = [i for i in doc.lower().split() if i not in stop]
            punc_free = [ch for ch in stop_free if ch not in exclude]
            normalized = [lemma.lemmatize(word) for word in punc_free]
            #normalized2 = [w for w in normalized if w != "would" and w != "people"]
        
            #yield [word for word in doc.lower().split() if word not in stop]
            yield  normalized


def get_dictionary(texts, min_count=1):
    """Construct dictionary 
    
    """
    dictionary = corpora.Dictionary(texts)
    lowfreq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() 
                    if docfreq < min_count]
    # remove stop words and low frequence words
    dictionary.filter_tokens(lowfreq_ids)
    # remove gaps in id sequence after words that were removed
    dictionary.compactify()
    
    #dictionary.save('docs.dict')
    return dictionary


def corpus2bow(texts,dictionary):
    """represent docs into a list with bag of words model
       bow: bag of words
    
    """
    corpus=[dictionary.doc2bow(text) for text in texts]
    
    return corpus

def bow2tfidf(corpus):
    """represent docs  with TF*IDF model
    
    """
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus] # wrap the old corpus to tfidf
    
    return corpus_tfidf
        
def topic_models(corpus,dictionary,num_topics=2, edges=None, labled_topics = None):
    """modelling the corpus with LDA, LSI and HDP
    
    """
    SOME_FIXED_SEED = 42
    #random_state = np.random.seed(SOME_FIXED_SEED)
    LDA_model = models.LdaModel(corpus = corpus, id2word = dictionary, num_topics=num_topics, edges=None)
    #LDA_model.save("3dtyke_raw")

    #LDA_model = models.LdaModel.load('3dtyke_raw')
    topics =  LDA_model.show_topics(num_words=10, log=False, formatted=True)
    for t in topics:
        print t
    

    #LDA_model.save("3dtyke_raw")
    '''
    i=0
    for c in corpus:
        doc_t =  LDA_model.get_document_topics(c)
        print i, doc_t
        i+=1
    '''
        
    '''
    topics_label= [0,1,2]

    matched=[]
    for i, c in enumerate(corpus):
        doc =  LDA_model.get_document_topics(c)
        topic_assign = max([d for d in doc], key=lambda item: (item[1]))
        topic_converted = topics_label[topic_assign[0]]
        if topic_converted == labled_topics[i]:
            matched.append(1)
        else:
            matched.append(0)
            
    print matched
    
    accuracy = float(sum(matched))/len(matched)
    print 'accuracy:',accuracy
    '''
        
    '''
    parent = LDA_model.get_document_topics(corpus[0])
    inherit = 0.3
    LDA_model.
    index = 0
    for c in corpus:
        doc_t =  LDA_model.get_document_topics(c)
        parent = tree.parent(index).identifier
        result = [(p[0], inherit * p[1]) for p in parent]
        for i in range(len(doc_t)):
            result[i] = ((doc_t[i][0],  (1-inherit) * doc_t[i][1] + inherit * parent[i][1]))
        
        print index, result
        index+=1  
    '''
 
        
    '''
    #Plot nodes to see clustering status
    
    nodes = list(LDA_model[corpus] )
    ax0 = [x[0][1] for x in nodes] 
    ax1 = [x[1][1] for x in nodes]
    
    plt.plot(ax0,ax1,'o')
    plt.show()
    
    '''

    
    return LDA_model

def word_distribution(texts):
  
    dict = {}
    for t in  texts:
        for word in t:
            if dict.has_key(word):
                dict[word] += 1
            else:
                dict[word] = 1
    print dict
    '''
    keys=[]
    values=[]

    for key,value in sorted(dict.iteritems(),key=lambda (k,v): (v,k), reverse  = True ):
        #print "%s: %s" % (key, value)
        keys.append(key)
        values.append(value) 
    
    
    x = range(len(keys))
    y = values
    
    plt.bar(x, y,  color='b') 
    plt.xticks(x, keys, rotation = 45)
    
    plt.savefig(file_dir+'\\graphs\\'+prefix_time()+file_name+'_word distribution'+'.png')
    plt.title('Word Distribution')
    plt.show()
    
    c = Counter(values)
    total = np.sum(np.array(c.values()))
    print total
    x= np.array(c.values())/float(total)

    labels1= c.keys()
    #s_length = len(x)
    #explode1=[0]*s_length
    plt.pie(x,autopct='%.0f%%',shadow=True,  labels=labels1, textprops = {'fontsize':14, 'color':'k'})
    plt.savefig(file_dir+'\\graphs\\'+prefix_time()+file_name+'_word frequency'+'.png')
    plt.title('Word Frequency')
    plt.show()
    '''

def node_distribution(nodes):
    nodes = sorted(nodes, reverse  =True)
    
    x = range(len(nodes))
    y = nodes

    plt.bar(x, y,  color='r') 
    plt.xticks(x, y, rotation = 45)
    #plt.ylim([0,110])
    plt.savefig(file_dir+'\\graphs\\'+prefix_time()+file_name+'_node distribution'+'.png')
    plt.title('Node Distribution')
    plt.show()
    
    dict = Counter(nodes)
    
    total = np.sum(np.array(dict.values()))
    print total
    x= np.array(dict.values())/float(total)

    labels1= dict.keys()
    #s_length = len(x)
    #explode1=[0]*s_length
    plt.pie(x,autopct='%.0f%%',shadow=True,  labels=labels1, textprops = {'fontsize':14, 'color':'k'})
    plt.savefig(file_dir+'\\graphs\\'+prefix_time()+file_name+'_node frequency'+'.png')
    plt.title('Node Frequency')
    plt.show()

def corpus_distribution(corpus):
    dict = {}
   
    for doc in  corpus:
        for word in doc:
            if dict.has_key(word[0]):
                dict[word[0]] += word[1]
            else:
                dict[word[0]] = word[1]
    keys=[]
    values=[]

    for key,value in sorted(dict.iteritems(),key=lambda (k,v): (v,k), reverse  = True ):
        #print "%s: %s" % (key, value)
        keys.append(key)
        values.append(value) 
            
    x = range(len(keys))
    y = values
    
    plt.bar(x, y,  color='b') 
    plt.xticks(x, keys, rotation = 45)
    #plt.ylim([0,500])
    
    plt.savefig(file_dir+'\\graphs\\'+prefix_time()+file_name+'_corpus distribution'+'.png')
    plt.title('Corpus Distribution')
    plt.show()
    
    c = Counter(values)
    total = np.sum(np.array(c.values()))

    x= np.array(c.values())/float(total)

    labels1= c.keys()
    #s_length = len(x)
    #explode1=[0]*s_length
    plt.pie(x,autopct='%.0f%%',shadow=True,  labels=labels1, textprops = {'fontsize':14, 'color':'k'})
    plt.savefig(file_dir+'\\graphs\\'+prefix_time()+file_name+'_corpus frequency'+'.png')
    plt.title('Corpus Frequency')
    plt.show()
    
if __name__ == '__main__':
    
    file_name = "8sk1ue"
    edges1, label1, documents1, labled_topics1 = file_reader(file_name)
    #edges2, label2, documents2 = file_reader("8sk1ue") 867njq  3dtyke
    
    simple2= ["lBJ Lakers ",
              "Warriors Championship",
              "money in the wallet to buy ticket",
              "Oh my gosh",
              "I will not watch Cavs's game",
              "Texas serial bomber made video confession before blowing himself up"
              ]
    #f=open("stemed.txt","w")
    texts = MyTexts(documents1)
    '''
    for t in texts:
        print >> f, " ".join(t)
       
    '''
    
    #word_distribution(texts)
   
    
    #tree_degree_height1, parents = get_degree_height(edges1, label1)
    
    #tree_degree_height2 = get_degree_height(edges2, label2)
    #train_with_degree_height(tree_degree_height1,label1)
    #cross_training(tree_degree_height1, label1, tree_degree_height2, label2)

    dictionary = get_dictionary(texts, min_count=1)
  
    # save and load dictionary
    '''
    dictionary.save('docs.dict')
    dictionary = corpora.Dictionary.load('docs.dict')
    print dictionary
    '''
    corpus = corpus2bow(texts,dictionary)
    
    
    corpus_tfidf = bow2tfidf(corpus)
    #doc="Human computer interaction"
    #print doc_similarity(doc, corpus)
    num_topics = 3
    
    magnification = 1
    #base too large is not good, it should be close to 0
    base = 0.000
    
    #867njq_new
    #node_degrees = [1.0, 0.99, 0.49, 0.23, 0.04, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.03, 0.02, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.48, 0.37, 0.33, 0.06, 0.01, 0.0, 0.03, 0.02, 0.01, 0.0, 0.0, 0.03, 0.01, 0.0, 0.0, 0.0, 0.03, 0.02, 0.01, 0.0, 0.06, 0.05, 0.04, 0.02, 0.01, 0.0, 0.0, 0.03, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.08, 0.05, 0.04, 0.01, 0.0, 0.0, 0.0, 0.01, 0.0]
    #node_heights = [1.0, 0.9, 0.6, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.6, 0.4, 0.1, 0.0, 0.3, 0.2, 0.1, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1, 0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0]
    
    #raw
    #node_degrees = [101, 100, 50, 24, 5, 1, 1, 1, 1, 11, 6, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 4, 3, 1, 1, 2, 1, 2, 1, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 49, 38, 34, 7, 2, 1, 4, 3, 2, 1, 1, 4, 2, 1, 1, 1, 4, 3, 2, 1, 7, 6, 5, 3, 2, 1, 1, 4, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 10, 9, 6, 5, 2, 1, 1, 1, 2, 1]
   
    #node_degrees = [101, 100, 50, 24, 5, 1, 1, 1, 1, 11, 6, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 4, 3, 1, 1, 2, 1, 2, 1, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 49, 38, 34, 7, 2, 1, 4, 3, 2, 1, 1, 4, 2, 1, 1, 1, 4, 3, 2, 1, 7, 6, 5, 3, 2, 1, 1, 4, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 10, 9, 6, 5, 2, 1, 1, 1, 2, 1]
    #node_heights = [11, 10, 7, 6, 2, 1, 1, 1, 1, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 2, 1, 1, 2, 1, 2, 1, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 8, 7, 5, 2, 1, 4, 3, 2, 1, 1, 3, 2, 1, 1, 1, 4, 3, 2, 1, 6, 5, 4, 3, 2, 1, 1, 3, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 6, 5, 4, 3, 2, 1, 1, 1, 2, 1]
    #H=11
    
    #single feature
    #node_degrees = [0.9, 0.892, 0.492, 0.28400000000000003, 0.132, 0.1, 0.1, 0.1, 0.1, 0.18000000000000002, 0.14, 0.124, 0.116, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.164, 0.1, 0.1, 0.124, 0.116, 0.1, 0.1, 0.10800000000000001, 0.1, 0.10800000000000001, 0.1, 0.124, 0.116, 0.10800000000000001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.484, 0.396, 0.364, 0.14800000000000002, 0.10800000000000001, 0.1, 0.124, 0.116, 0.10800000000000001, 0.1, 0.1, 0.124, 0.10800000000000001, 0.1, 0.1, 0.1, 0.124, 0.116, 0.10800000000000001, 0.1, 0.14800000000000002, 0.14, 0.132, 0.116, 0.10800000000000001, 0.1, 0.1, 0.124, 0.10800000000000001, 0.1, 0.1, 0.10800000000000001, 0.1, 0.10800000000000001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.172, 0.164, 0.14, 0.132, 0.10800000000000001, 0.1, 0.1, 0.1, 0.10800000000000001, 0.1]
    #node_degrees = [1.0, 0.99, 0.49, 0.23, 0.04, 0.0, 0.0, 0.0, 0.0, 0.1, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.03, 0.02, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.48, 0.37, 0.33, 0.06, 0.01, 0.0, 0.03, 0.02, 0.01, 0.0, 0.0, 0.03, 0.01, 0.0, 0.0, 0.0, 0.03, 0.02, 0.01, 0.0, 0.06, 0.05, 0.04, 0.02, 0.01, 0.0, 0.0, 0.03, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09, 0.08, 0.05, 0.04, 0.01, 0.0, 0.0, 0.0, 0.01, 0.0]
    #node_heights = [1.0, 0.9, 0.6, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.6, 0.4, 0.1, 0.0, 0.3, 0.2, 0.1, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1, 0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0]
    
    #scaled_weigths
    #node_degrees = [30.452645802354127, 31.622776601683793, 18.89822365046136, 9.797958971132713, 3.5355339059327373, 1.0, 1.0, 1.0, 1.0, 4.919349550499537, 3.0, 2.3094010767585034, 2.1213203435596424, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.5, 1.0, 1.0, 2.3094010767585034, 2.1213203435596424, 1.0, 1.0, 1.414213562373095, 1.0, 1.414213562373095, 1.0, 2.0, 1.7320508075688774, 1.414213562373095, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 16.333333333333332, 13.435028842544401, 12.850792082313726, 3.1304951684997055, 1.414213562373095, 1.0, 2.0, 1.7320508075688774, 1.414213562373095, 1.0, 1.0, 2.3094010767585034, 1.414213562373095, 1.0, 1.0, 1.0, 2.0, 1.7320508075688774, 1.414213562373095, 1.0, 2.8577380332470415, 2.6832815729997477, 2.5, 1.7320508075688774, 1.414213562373095, 1.0, 1.0, 2.3094010767585034, 1.414213562373095, 1.0, 1.0, 1.414213562373095, 1.0, 1.414213562373095, 1.0, 1.0, 1.0, 1.0, 1.0, 4.08248290463863, 4.024922359499621, 3.0, 2.886751345948129, 1.414213562373095, 1.0, 1.0, 1.0, 1.414213562373095, 1.0]
    #node_degrees = [10.04987562112089, 10.0, 7.0710678118654755, 4.898979485566356, 2.23606797749979, 1.0, 1.0, 1.0, 1.0, 3.3166247903554, 2.449489742783178, 2.0, 1.7320508075688772, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.7320508075688772, 1.0, 1.0, 1.4142135623730951, 1.0, 1.4142135623730951, 1.0, 2.0, 1.7320508075688772, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0, 6.164414002968976, 5.830951894845301, 2.6457513110645907, 1.4142135623730951, 1.0, 2.0, 1.7320508075688772, 1.4142135623730951, 1.0, 1.0, 2.0, 1.4142135623730951, 1.0, 1.0, 1.0, 2.0, 1.7320508075688772, 1.4142135623730951, 1.0, 2.6457513110645907, 2.449489742783178, 2.23606797749979, 1.7320508075688772, 1.4142135623730951, 1.0, 1.0, 2.0, 1.4142135623730951, 1.0, 1.0, 1.4142135623730951, 1.0, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 3.1622776601683795, 3.0, 2.449489742783178, 2.23606797749979, 1.4142135623730951, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0]
    #node_degrees = [1.0, 0.9950371902099892, 0.7035975447302919, 0.48746667822143247, 0.22249707974499242, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.3300165012376031, 0.24373333911071624, 0.19900743804199783, 0.1723454968864278, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.29851115706299675, 0.09950371902099892, 0.09950371902099892, 0.19900743804199783, 0.1723454968864278, 0.09950371902099892, 0.09950371902099892, 0.1407195089460584, 0.09950371902099892, 0.1407195089460584, 0.09950371902099892, 0.19900743804199783, 0.1723454968864278, 0.1407195089460584, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.6965260331469925, 0.6133821188805362, 0.5802013989696481, 0.26326209505561055, 0.1407195089460584, 0.09950371902099892, 0.19900743804199783, 0.1723454968864278, 0.1407195089460584, 0.09950371902099892, 0.09950371902099892, 0.19900743804199783, 0.1407195089460584, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.19900743804199783, 0.1723454968864278, 0.1407195089460584, 0.09950371902099892, 0.26326209505561055, 0.24373333911071624, 0.22249707974499242, 0.1723454968864278, 0.1407195089460584, 0.09950371902099892, 0.09950371902099892, 0.19900743804199783, 0.1407195089460584, 0.09950371902099892, 0.09950371902099892, 0.1407195089460584, 0.09950371902099892, 0.1407195089460584, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.31465838776377636, 0.29851115706299675, 0.24373333911071624, 0.22249707974499242, 0.1407195089460584, 0.09950371902099892, 0.09950371902099892, 0.09950371902099892, 0.1407195089460584, 0.09950371902099892]
    
    #popularity
    #node_degrees = [54.72727272727274, 59.1, 33.285714285714285, 16.166666666666668, 3.0, 1.0, 1.0, 1.0, 1.0, 7.0, 3.5, 2.333333333333333, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.5, 1.0, 1.0, 2.333333333333333, 2.0, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 2.5, 1.9999999999999998, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 27.555555555555564, 23.375, 22.571428571428573, 4.4, 1.5, 1.0, 2.5, 1.9999999999999998, 1.5, 1.0, 1.0, 2.6666666666666665, 1.5, 1.0, 1.0, 1.0, 2.5, 1.9999999999999998, 1.5, 1.0, 4.0, 3.6, 3.25, 1.9999999999999998, 1.5, 1.0, 1.0, 2.6666666666666665, 1.5, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 5.333333333333334, 5.199999999999999, 3.5, 3.333333333333333, 1.5, 1.0, 1.0, 1.0, 1.5, 1.0]
    
    #Arithmetic Progression
    #node_degrees = [54.72727272727274, 62.81818181818179, 39.36363636363633, 19.72727272727273, 4.636363636363637, 1.0, 1.0, 1.0, 1.0, 9.181818181818182, 5.090909090909092, 3.545454545454546, 2.8181818181818183, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.727272727272728, 1.0, 1.0, 3.545454545454546, 2.8181818181818183, 1.0, 1.0, 1.9090909090909092, 1.0, 1.9090909090909092, 1.0, 3.454545454545455, 2.7272727272727275, 1.9090909090909092, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 31.45454545454545, 27.363636363636356, 26.727272727272723, 5.818181818181819, 1.9090909090909092, 1.0, 3.454545454545455, 2.7272727272727275, 1.9090909090909092, 1.0, 1.0, 3.6363636363636367, 1.9090909090909092, 1.0, 1.0, 1.0, 3.454545454545455, 2.7272727272727275, 1.9090909090909092, 1.0, 5.363636363636364, 4.90909090909091, 4.363636363636364, 2.7272727272727275, 1.9090909090909092, 1.0, 1.0, 3.6363636363636367, 1.9090909090909092, 1.0, 1.0, 1.9090909090909092, 1.0, 1.9090909090909092, 1.0, 1.0, 1.0, 1.0, 1.0, 7.454545454545456, 7.272727272727274, 5.090909090909092, 4.545454545454546, 1.9090909090909092, 1.0, 1.0, 1.0, 1.9090909090909092, 1.0]
    
    #Geometric Progression
    #node_degrees = [35.43213475839999, 43.040168448, 30.84396800000002, 16.048960000000008, 4.2, 1.0, 1.0, 1.0, 1.0, 7.6112, 4.264, 3.0800000000000005, 2.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.504, 1.0, 1.0, 3.0800000000000005, 2.6, 1.0, 1.0, 1.8, 1.0, 1.8, 1.0, 2.9520000000000004, 2.4400000000000004, 1.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 21.706242560000007, 20.222323200000012, 21.02790400000001, 4.8016000000000005, 1.8, 1.0, 2.9520000000000004, 2.4400000000000004, 1.8, 1.0, 1.0, 3.24, 1.8, 1.0, 1.0, 1.0, 2.9520000000000004, 2.4400000000000004, 1.8, 1.0, 4.201280000000001, 4.001600000000001, 3.7520000000000007, 2.4400000000000004, 1.8, 1.0, 1.0, 3.24, 1.8, 1.0, 1.0, 1.8, 1.0, 1.8, 1.0, 1.0, 1.0, 1.0, 1.0, 5.6604800000000015, 5.825600000000001, 4.232000000000001, 4.04, 1.8, 1.0, 1.0, 1.0, 1.8, 1.0]
    
    #Harmonic Progression
    #node_degrees = [18.913924963924956, 22.592460317460322, 17.452380952380953, 9.533333333333333, 3.0, 1.0, 1.0, 1.0, 1.0, 4.816666666666666, 2.833333333333333, 2.1666666666666665, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.166666666666666, 1.0, 1.0, 2.1666666666666665, 2.0, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 2.083333333333333, 1.8333333333333333, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 11.642063492063492, 11.110714285714286, 12.109523809523811, 3.1166666666666667, 1.5, 1.0, 2.083333333333333, 1.8333333333333333, 1.5, 1.0, 1.0, 2.333333333333333, 1.5, 1.0, 1.0, 1.0, 2.083333333333333, 1.8333333333333333, 1.5, 1.0, 2.6999999999999997, 2.6166666666666667, 2.583333333333333, 1.8333333333333333, 1.5, 1.0, 1.0, 2.333333333333333, 1.5, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 3.4333333333333336, 3.6166666666666667, 2.75, 2.833333333333333, 1.5, 1.0, 1.0, 1.0, 1.5, 1.0]

    #8sk1ue
    #node_degrees = [1, 0.58, 0.53, 0.11, 0.0, 0.04, 0.03, 0.02, 0.01, 0.0, 0.03, 0.0, 0.01, 0.0, 0.0, 0.36, 0.11, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0, 0.0, 0.02, 0.0, 0.0, 0.22, 0.12, 0.11, 0.07, 0.01, 0.0, 0.02, 0.01, 0.0, 0.01, 0.0, 0.0, 0.01, 0.0, 0.08, 0.05, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.39, 0.0, 0.36, 0.16, 0.15, 0.14, 0.13, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.02, 0.0, 0.13, 0.0, 0.04, 0.03, 0.02, 0.01, 0.0, 0.03, 0.02, 0.0, 0.0, 0.02, 0.01, 0.0, 0.0, 0.0]
    #node_heights = [1, 0.5882352941176471, 0.5294117647058824, 0.29411764705882354, 0.0, 0.23529411764705882, 0.17647058823529413, 0.11764705882352941, 0.058823529411764705, 0.0, 0.11764705882352941, 0.0, 0.058823529411764705, 0.0, 0.0, 0.47058823529411764, 0.4117647058823529, 0.35294117647058826, 0.29411764705882354, 0.23529411764705882, 0.17647058823529413, 0.11764705882352941, 0.058823529411764705, 0.0, 0.0, 0.058823529411764705, 0.0, 0.0, 0.35294117647058826, 0.29411764705882354, 0.23529411764705882, 0.17647058823529413, 0.058823529411764705, 0.0, 0.11764705882352941, 0.058823529411764705, 0.0, 0.058823529411764705, 0.0, 0.0, 0.058823529411764705, 0.0, 0.17647058823529413, 0.11764705882352941, 0.0, 0.058823529411764705, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.058823529411764705, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9411764705882353, 0.0, 0.8823529411764706, 0.8235294117647058, 0.7647058823529411, 0.7058823529411765, 0.6470588235294118, 0.5882352941176471, 0.5294117647058824, 0.47058823529411764, 0.4117647058823529, 0.35294117647058826, 0.29411764705882354, 0.23529411764705882, 0.17647058823529413, 0.11764705882352941, 0.058823529411764705, 0.0, 0.058823529411764705, 0.0, 0.058823529411764705, 0.0, 0.058823529411764705, 0.0, 0.29411764705882354, 0.0, 0.23529411764705882, 0.17647058823529413, 0.11764705882352941, 0.058823529411764705, 0.0, 0.11764705882352941, 0.058823529411764705, 0.0, 0.0, 0.11764705882352941, 0.058823529411764705, 0.0, 0.0, 0.0]
    
    #popularity
    #node_degrees = [67.33333333333334, 33.454545454545446, 31.699999999999996, 7.833333333333334, 1.0, 3.0, 2.5, 1.9999999999999998, 1.5, 1.0, 2.6666666666666665, 1.0, 1.5, 1.0, 1.0, 21.000000000000007, 7.625, 4.857142857142857, 3.5, 3.0, 2.5, 1.9999999999999998, 1.5, 1.0, 1.0, 2.0, 1.0, 1.0, 12.57142857142857, 7.000000000000001, 7.199999999999999, 5.0, 1.5, 1.0, 1.9999999999999998, 1.5, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 5.25, 3.6666666666666665, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 26.764705882352953, 1.0, 25.375, 9.266666666666666, 8.857142857142856, 8.461538461538462, 8.083333333333332, 7.727272727272727, 5.5, 4.999999999999999, 4.5, 4.0, 3.5, 3.0, 2.5, 1.9999999999999998, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 2.0, 1.0, 8.833333333333334, 1.0, 3.0, 2.5, 1.9999999999999998, 1.5, 1.0, 2.333333333333333, 2.0, 1.0, 1.0, 1.9999999999999998, 1.5, 1.0, 1.0, 1.0]

    #0.1-0.9
    #node_degrees = [0.9, 0.564, 0.524, 0.188, 0.1, 0.132, 0.124, 0.116, 0.10800000000000001, 0.1, 0.124, 0.1, 0.10800000000000001, 0.1, 0.1, 0.388, 0.188, 0.15600000000000003, 0.14, 0.132, 0.124, 0.116, 0.10800000000000001, 0.1, 0.1, 0.116, 0.1, 0.1, 0.276, 0.196, 0.188, 0.15600000000000003, 0.10800000000000001, 0.1, 0.116, 0.10800000000000001, 0.1, 0.10800000000000001, 0.1, 0.1, 0.10800000000000001, 0.1, 0.164, 0.14, 0.1, 0.116, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.10800000000000001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.41200000000000003, 0.1, 0.388, 0.228, 0.22, 0.21200000000000002, 0.20400000000000001, 0.196, 0.172, 0.164, 0.15600000000000003, 0.14800000000000002, 0.14, 0.132, 0.124, 0.116, 0.10800000000000001, 0.1, 0.10800000000000001, 0.1, 0.10800000000000001, 0.1, 0.116, 0.1, 0.20400000000000001, 0.1, 0.132, 0.124, 0.116, 0.10800000000000001, 0.1, 0.124, 0.116, 0.1, 0.1, 0.116, 0.10800000000000001, 0.1, 0.1, 0.1]
    #node_heights = [0.9, 0.5705882352941177, 0.5235294117647059, 0.33529411764705885, 0.1, 0.28823529411764703, 0.24117647058823533, 0.19411764705882353, 0.14705882352941177, 0.1, 0.19411764705882353, 0.1, 0.14705882352941177, 0.1, 0.1, 0.4764705882352941, 0.4294117647058824, 0.3823529411764707, 0.33529411764705885, 0.28823529411764703, 0.24117647058823533, 0.19411764705882353, 0.14705882352941177, 0.1, 0.1, 0.14705882352941177, 0.1, 0.1, 0.3823529411764707, 0.33529411764705885, 0.28823529411764703, 0.24117647058823533, 0.14705882352941177, 0.1, 0.19411764705882353, 0.14705882352941177, 0.1, 0.14705882352941177, 0.1, 0.1, 0.14705882352941177, 0.1, 0.24117647058823533, 0.19411764705882353, 0.1, 0.14705882352941177, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.14705882352941177, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8529411764705882, 0.1, 0.8058823529411765, 0.7588235294117647, 0.711764705882353, 0.6647058823529413, 0.6176470588235294, 0.5705882352941177, 0.5235294117647059, 0.4764705882352941, 0.4294117647058824, 0.3823529411764707, 0.33529411764705885, 0.28823529411764703, 0.24117647058823533, 0.19411764705882353, 0.14705882352941177, 0.1, 0.14705882352941177, 0.1, 0.14705882352941177, 0.1, 0.14705882352941177, 0.1, 0.33529411764705885, 0.1, 0.28823529411764703, 0.24117647058823533, 0.19411764705882353, 0.14705882352941177, 0.1, 0.19411764705882353, 0.14705882352941177, 0.1, 0.1, 0.19411764705882353, 0.14705882352941177, 0.1, 0.1, 0.1]
    
    #raw data
    #node_degrees =[101, 59, 54, 12, 1, 5, 4, 3, 2, 1, 4, 1, 2, 1, 1, 37, 12, 8, 6, 5, 4, 3, 2, 1, 1, 3, 1, 1, 23, 13, 12, 8, 2, 1, 3, 2, 1, 2, 1, 1, 2, 1, 9, 6, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 40, 1, 37, 17, 16, 15, 14, 13, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 1, 2, 1, 3, 1, 14, 1, 5, 4, 3, 2, 1, 4, 3, 1, 1, 3, 2, 1, 1, 1]
    #node_heights =[18, 11, 10, 6, 1, 5, 4, 3, 2, 1, 3, 1, 2, 1, 1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 1, 1, 7, 6, 5, 4, 2, 1, 3, 2, 1, 2, 1, 1, 2, 1, 4, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 17, 1, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 1, 2, 1, 2, 1, 6, 1, 5, 4, 3, 2, 1, 3, 2, 1, 1, 3, 2, 1, 1, 1]
    #H=18
    
    #node_degrees = [1.0, 0.5841584158415841, 0.5346534653465347, 0.1188118811881188, 0.009900990099009901, 0.04950495049504951, 0.039603960396039604, 0.0297029702970297, 0.019801980198019802, 0.009900990099009901, 0.039603960396039604, 0.009900990099009901, 0.019801980198019802, 0.009900990099009901, 0.009900990099009901, 0.36633663366336633, 0.1188118811881188, 0.07920792079207921, 0.0594059405940594, 0.04950495049504951, 0.039603960396039604, 0.0297029702970297, 0.019801980198019802, 0.009900990099009901, 0.009900990099009901, 0.0297029702970297, 0.009900990099009901, 0.009900990099009901, 0.22772277227722773, 0.12871287128712872, 0.1188118811881188, 0.07920792079207921, 0.019801980198019802, 0.009900990099009901, 0.0297029702970297, 0.019801980198019802, 0.009900990099009901, 0.019801980198019802, 0.009900990099009901, 0.009900990099009901, 0.019801980198019802, 0.009900990099009901, 0.0891089108910891, 0.0594059405940594, 0.009900990099009901, 0.0297029702970297, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.019801980198019802, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901, 0.39603960396039606, 0.009900990099009901, 0.36633663366336633, 0.16831683168316833, 0.15841584158415842, 0.1485148514851485, 0.13861386138613863, 0.12871287128712872, 0.09900990099009901, 0.0891089108910891, 0.07920792079207921, 0.06930693069306931, 0.0594059405940594, 0.04950495049504951, 0.039603960396039604, 0.0297029702970297, 0.019801980198019802, 0.009900990099009901, 0.019801980198019802, 0.009900990099009901, 0.019801980198019802, 0.009900990099009901, 0.0297029702970297, 0.009900990099009901, 0.13861386138613863, 0.009900990099009901, 0.04950495049504951, 0.039603960396039604, 0.0297029702970297, 0.019801980198019802, 0.009900990099009901, 0.039603960396039604, 0.0297029702970297, 0.009900990099009901, 0.009900990099009901, 0.0297029702970297, 0.019801980198019802, 0.009900990099009901, 0.009900990099009901, 0.009900990099009901]
    #node_heights = [1.0, 0.6111111111111112, 0.5555555555555556, 0.3333333333333333, 0.05555555555555555, 0.2777777777777778, 0.2222222222222222, 0.16666666666666666, 0.1111111111111111, 0.05555555555555555, 0.16666666666666666, 0.05555555555555555, 0.1111111111111111, 0.05555555555555555, 0.05555555555555555, 0.5, 0.4444444444444444, 0.3888888888888889, 0.3333333333333333, 0.2777777777777778, 0.2222222222222222, 0.16666666666666666, 0.1111111111111111, 0.05555555555555555, 0.05555555555555555, 0.1111111111111111, 0.05555555555555555, 0.05555555555555555, 0.3888888888888889, 0.3333333333333333, 0.2777777777777778, 0.2222222222222222, 0.1111111111111111, 0.05555555555555555, 0.16666666666666666, 0.1111111111111111, 0.05555555555555555, 0.1111111111111111, 0.05555555555555555, 0.05555555555555555, 0.1111111111111111, 0.05555555555555555, 0.2222222222222222, 0.16666666666666666, 0.05555555555555555, 0.1111111111111111, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.1111111111111111, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.9444444444444444, 0.05555555555555555, 0.8888888888888888, 0.8333333333333334, 0.7777777777777778, 0.7222222222222222, 0.6666666666666666, 0.6111111111111112, 0.5555555555555556, 0.5, 0.4444444444444444, 0.3888888888888889, 0.3333333333333333, 0.2777777777777778, 0.2222222222222222, 0.16666666666666666, 0.1111111111111111, 0.05555555555555555, 0.1111111111111111, 0.05555555555555555, 0.1111111111111111, 0.05555555555555555, 0.1111111111111111, 0.05555555555555555, 0.3333333333333333, 0.05555555555555555, 0.2777777777777778, 0.2222222222222222, 0.16666666666666666, 0.1111111111111111, 0.05555555555555555, 0.16666666666666666, 0.1111111111111111, 0.05555555555555555, 0.05555555555555555, 0.16666666666666666, 0.1111111111111111, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555]
    
    #scaled_weigths
    #node_degrees = [23.8059282999471, 17.789169330088054, 17.076299364909246, 4.898979485566357, 1.0, 2.23606797749979, 2.0, 1.7320508075688774, 1.414213562373095, 1.0, 2.3094010767585034, 1.0, 1.414213562373095, 1.0, 1.0, 12.333333333333334, 4.242640687119285, 3.0237157840738176, 2.4494897427831783, 2.23606797749979, 2.0, 1.7320508075688774, 1.414213562373095, 1.0, 1.0, 2.1213203435596424, 1.0, 1.0, 8.693182879212225, 5.30722777603022, 5.366563145999495, 4.0, 1.414213562373095, 1.0, 1.7320508075688774, 1.414213562373095, 1.0, 1.414213562373095, 1.0, 1.0, 1.414213562373095, 1.0, 4.5, 3.464101615137755, 1.0, 2.1213203435596424, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.414213562373095, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.70142500145332, 1.0, 9.25, 4.389381125701739, 4.27617987059879, 4.160251471689219, 4.041451884327381, 3.9196474795109273, 3.162277660168379, 3.0, 2.82842712474619, 2.6457513110645903, 2.4494897427831783, 2.23606797749979, 2.0, 1.7320508075688774, 1.414213562373095, 1.0, 1.414213562373095, 1.0, 1.414213562373095, 1.0, 2.1213203435596424, 1.0, 5.715476066494083, 1.0, 2.23606797749979, 2.0, 1.7320508075688774, 1.414213562373095, 1.0, 2.3094010767585034, 2.1213203435596424, 1.0, 1.0, 1.7320508075688774, 1.414213562373095, 1.0, 1.0, 1.0]

    #Arithmetic Progression
    #node_degrees = [67.33333333333334, 43.388888888888886, 41.61111111111111, 10.611111111111112, 1.0, 4.444444444444445, 3.6666666666666665, 2.833333333333333, 1.9444444444444444, 1.0, 3.7777777777777777, 1.0, 1.9444444444444444, 1.0, 1.0, 28.999999999999996, 10.055555555555557, 6.777777777777779, 5.166666666666667, 4.444444444444445, 3.6666666666666665, 2.833333333333333, 1.9444444444444444, 1.0, 1.0, 2.888888888888889, 1.0, 1.0, 18.94444444444445, 11.000000000000002, 10.666666666666666, 7.333333333333332, 1.9444444444444444, 1.0, 2.833333333333333, 1.9444444444444444, 1.0, 1.9444444444444444, 1.0, 1.0, 1.9444444444444444, 1.0, 8.166666666666666, 5.611111111111111, 1.0, 2.888888888888889, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.9444444444444444, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 27.499999999999996, 1.0, 26.666666666666668, 10.555555555555555, 10.444444444444446, 10.27777777777778, 10.055555555555557, 9.777777777777779, 7.5, 7.0, 6.444444444444445, 5.833333333333334, 5.166666666666667, 4.444444444444445, 3.6666666666666665, 2.833333333333333, 1.9444444444444444, 1.0, 1.9444444444444444, 1.0, 1.9444444444444444, 1.0, 2.888888888888889, 1.0, 12.277777777777779, 1.0, 4.444444444444445, 3.6666666666666665, 2.833333333333333, 1.9444444444444444, 1.0, 3.722222222222222, 2.888888888888889, 1.0, 1.0, 2.833333333333333, 1.9444444444444444, 1.0, 1.0, 1.0]

    #Geometric Progression
    #node_degrees = [32.7136387581726, 23.261771878400005, 23.827214848000008, 7.881280000000001, 1.0, 3.3616000000000006, 2.9520000000000004, 2.4400000000000004, 1.8, 1.0, 3.24, 1.0, 1.8, 1.0, 1.0, 16.852738560000002, 6.881139200000002, 4.751424000000001, 3.6892800000000006, 3.3616000000000006, 2.9520000000000004, 2.4400000000000004, 1.8, 1.0, 1.0, 2.6, 1.0, 1.0, 11.934784000000002, 7.324480000000003, 7.9056000000000015, 5.832000000000001, 1.8, 1.0, 2.4400000000000004, 1.8, 1.0, 1.8, 1.0, 1.0, 1.8, 1.0, 6.344000000000001, 4.6800000000000015, 1.0, 2.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 15.380276569315745, 1.0, 15.975345711644675, 5.413902139555841, 5.517377674444801, 5.646722093056002, 5.808402616320002, 6.010503270400001, 4.463129088000001, 4.328911360000001, 4.161139200000001, 3.9514240000000007, 3.6892800000000006, 3.3616000000000006, 2.9520000000000004, 2.4400000000000004, 1.8, 1.0, 1.8, 1.0, 1.8, 1.0, 2.6, 1.0, 8.905280000000001, 1.0, 3.3616000000000006, 2.9520000000000004, 2.4400000000000004, 1.8, 1.0, 3.0800000000000005, 2.6, 1.0, 1.0, 2.4400000000000004, 1.8, 1.0, 1.0, 1.0]

    #Harmonic Progression
    #node_degrees = [18.496370704458933, 13.11154401154401, 13.215079365079358, 4.866666666666667, 1.0, 2.283333333333333, 2.083333333333333, 1.8333333333333333, 1.5, 1.0, 2.3333333333333335, 1.0, 1.5, 1.0, 1.0, 9.332539682539684, 4.2178571428571425, 3.0928571428571425, 2.4499999999999997, 2.283333333333333, 2.083333333333333, 1.8333333333333333, 1.5, 1.0, 1.0, 2.0, 1.0, 1.0, 6.676190476190476, 4.266666666666667, 4.783333333333333, 3.75, 1.5, 1.0, 1.8333333333333333, 1.5, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 4.0, 3.166666666666667, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.96018744327568, 1.0, 9.341443278943277, 3.627752802752803, 3.6182289932289935, 3.6301337551337554, 3.6865440115440116, 3.853210678210678, 2.9289682539682538, 2.8289682539682537, 2.7178571428571425, 2.5928571428571425, 2.4499999999999997, 2.283333333333333, 2.083333333333333, 1.8333333333333333, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 2.0, 1.0, 5.366666666666666, 1.0, 2.283333333333333, 2.083333333333333, 1.8333333333333333, 1.5, 1.0, 2.1666666666666665, 2.0, 1.0, 1.0, 1.8333333333333333, 1.5, 1.0, 1.0, 1.0]

    #3dtyke_new
    #node_degrees = [1.0, 0.99, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.54, 0.43, 0.27, 0.02, 0.01, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.04, 0.03, 0.02, 0.01, 0.0, 0.09, 0.02, 0.01, 0.0, 0.01, 0.0, 0.03, 0.02, 0.0, 0.01, 0.0, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07, 0.06, 0.03, 0.02, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.21, 0.08, 0.02, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #node_heights = [1.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.6, 0.2, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.4, 0.2, 0.1, 0.0, 0.1, 0.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.3, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    #raw data
    #node_degrees = [101, 100, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 55, 44, 28, 3, 2, 1, 6, 1, 1, 1, 1, 1, 1, 11, 5, 4, 3, 2, 1, 10, 3, 2, 1, 2, 1, 4, 3, 1, 2, 1, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 8, 7, 4, 3, 1, 1, 2, 1, 2, 1, 22, 9, 3, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 5, 4, 3, 1, 1, 1, 1, 1, 1]
    #node_heights = [11, 10, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 8, 7, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 6, 5, 4, 3, 2, 1, 5, 3, 2, 1, 2, 1, 4, 3, 1, 2, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 5, 4, 3, 2, 1, 1, 2, 1, 2, 1, 4, 3, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 1, 1]
    #H = 11
    #parents = [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 14, 15, 16, 17, 18, 16, 20, 20, 20, 20, 20, 16, 16, 16, 28, 29, 30, 31, 27, 33, 34, 35, 33, 37, 33, 39, 16, 40, 42, 15, 44, 45, 45, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14, 14, 14, 61, 62, 14, 14, 14, 14, 14, 1, 69, 70, 71, 72, 72, 70, 75, 1, 77, 1, 79, 80, 81, 81, 80, 84, 84, 84, 84, 79, 79, 79, 79, 79, 93, 94, 94, 92, 92, 92, 92]
    
    #scaled_weights
    #node_degrees = [30.452645802354127, 31.622776601683793, 8.48528137423857, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 18.333333333333332, 15.556349186104045, 10.583005244258361, 1.7320508075688774, 1.414213562373095, 1.0, 4.242640687119285, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.4907311951024935, 2.23606797749979, 2.0, 1.7320508075688774, 1.414213562373095, 1.0, 4.47213595499958, 1.7320508075688774, 1.414213562373095, 1.0, 1.414213562373095, 1.0, 2.0, 1.7320508075688774, 1.0, 1.414213562373095, 1.0, 2.3094010767585034, 2.1213203435596424, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.7320508075688774, 1.414213562373095, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.5777087639996634, 3.5, 2.3094010767585034, 2.1213203435596424, 1.0, 1.0, 1.414213562373095, 1.0, 1.414213562373095, 1.0, 11.0, 5.196152422706632, 2.1213203435596424, 1.0, 1.0, 3.5355339059327373, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.5355339059327373, 2.3094010767585034, 2.1213203435596424, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    #scaled_degree
    #node_degrees = [10.04987562112089, 10.0, 3.4641016151377544, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.416198487095663, 6.6332495807108, 5.291502622129181, 1.7320508075688772, 1.4142135623730951, 1.0, 2.449489742783178, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.3166247903554, 2.23606797749979, 2.0, 1.7320508075688772, 1.4142135623730951, 1.0, 3.1622776601683795, 1.7320508075688772, 1.4142135623730951, 1.0, 1.4142135623730951, 1.0, 2.0, 1.7320508075688772, 1.0, 1.4142135623730951, 1.0, 2.0, 1.7320508075688772, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.7320508075688772, 1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.8284271247461903, 2.6457513110645907, 2.0, 1.7320508075688772, 1.0, 1.0, 1.4142135623730951, 1.0, 1.4142135623730951, 1.0, 4.69041575982343, 3.0, 1.7320508075688772, 1.0, 1.0, 2.23606797749979, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.23606797749979, 2.0, 1.7320508075688772, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    #popularity 
    #node_degrees = [59.8181818181818, 64.7, 6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 35.111111111111114, 28.75, 17.428571428571427, 1.9999999999999998, 1.5, 1.0, 3.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.166666666666667, 3.0, 2.5, 1.9999999999999998, 1.5, 1.0, 6.199999999999999, 1.9999999999999998, 1.5, 1.0, 1.5, 1.0, 2.5, 1.9999999999999998, 1.0, 1.5, 1.0, 2.333333333333333, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.9999999999999998, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.2, 4.0, 2.333333333333333, 2.0, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 11.0, 4.333333333333333, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 2.333333333333333, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    #Arithmetic Progression
    #node_degrees = [59.8181818181818, 67.90909090909092, 10.999999999999998, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 38.7272727272727, 32.909090909090914, 21.27272727272727, 2.7272727272727275, 1.9090909090909092, 1.0, 5.545454545454546, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.363636363636365, 4.090909090909092, 3.454545454545455, 2.7272727272727275, 1.9090909090909092, 1.0, 8.272727272727273, 2.7272727272727275, 1.9090909090909092, 1.0, 1.9090909090909092, 1.0, 3.454545454545455, 2.7272727272727275, 1.0, 1.9090909090909092, 1.0, 3.545454545454546, 2.8181818181818183, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.7272727272727275, 1.9090909090909092, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.272727272727274, 5.90909090909091, 3.545454545454546, 2.8181818181818183, 1.0, 1.0, 1.9090909090909092, 1.0, 1.9090909090909092, 1.0, 17.999999999999993, 7.727272727272728, 2.8181818181818183, 1.0, 1.0, 4.636363636363637, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.636363636363637, 3.545454545454546, 2.8181818181818183, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    #Geometric Progression
    #node_degrees = [40.03046000640001, 48.788075008, 9.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 28.835893760000015, 25.35486720000001, 16.363584000000007, 2.4400000000000004, 1.8, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.4028800000000015, 3.3616000000000006, 2.9520000000000004, 2.4400000000000004, 1.8, 1.0, 6.753600000000001, 2.4400000000000004, 1.8, 1.0, 1.8, 1.0, 2.9520000000000004, 2.4400000000000004, 1.0, 1.8, 1.0, 3.0800000000000005, 2.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.4400000000000004, 1.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.923200000000001, 4.904, 3.0800000000000005, 2.6, 1.0, 1.0, 1.8, 1.0, 1.8, 1.0, 14.376000000000008, 6.440000000000003, 2.6, 1.0, 1.0, 4.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.2, 3.0800000000000005, 2.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    #Harmonic Progression
    #node_degrees = [20.941305916305907, 25.830952380952358, 6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 16.05753968253969, 14.553571428571432, 9.359523809523807, 1.8333333333333333, 1.5, 1.0, 3.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.816666666666667, 2.283333333333333, 2.083333333333333, 1.8333333333333333, 1.5, 1.0, 4.2, 1.8333333333333333, 1.5, 1.0, 1.5, 1.0, 2.083333333333333, 1.8333333333333333, 1.0, 1.5, 1.0, 2.1666666666666665, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.8333333333333333, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.066666666666667, 3.1666666666666665, 2.1666666666666665, 2.0, 1.0, 1.0, 1.5, 1.0, 1.5, 1.0, 8.333333333333332, 4.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 2.1666666666666665, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    
    #node_distribution(node_degrees)
    #node_distribution(node_heights)
    
    
    #extension = 0.55*np.array(node_degrees) + 0.45*np.array(node_heights)
    #extension = np.array(node_degrees)
    #extension = [1,0.8,0.9,0,0,0]
    
    # for plotting the weight (structural part)
    '''
    extension_dist = []
    for i in range(len(extension)):
        #extension_dist += [doc[1]*(1 * magnification*(base +extension[i])) for doc in corpus[i]]
        
        extension_dist += [doc[1] for doc in corpus[i]]
    print extension_dist
    #extension_dist = [101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0, 50.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 6.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    node_distribution(extension_dist)
    '''

   
    #for i in range(len(extension)):
    #    corpus[i] = [(doc[0], doc[1]*(1 * magnification*(base +extension[i]))) for doc in corpus[i]]

    #corpus_distribution(corpus)
    
    lda_model = topic_models(corpus=corpus, dictionary=dictionary,num_topics=num_topics,edges=edges,labled_topics = labled_topics1)
  

    '''
    topic_comment = []
    for comment in corpus:
        topic_comment.append(lda_model.get_document_topics(comment))
    
    inherit = 0.3
    
    result = []
    index = 0
    for index in range(len(topic_comment)):
        parent = topic_comment[parents[index]]
        
        result = [(p[0], inherit * p[1]) for p in parent]
        for i in range(len(topic_comment[index])):
            result[i] = ((topic_comment[index][i][0],  (1-inherit) * topic_comment[index][i][1] + inherit * parent[i][1]))
          
        print index, topic_comment[index]
        print index, result
         
        topic_comment[index] = result
    '''
    #ldamodel_path = 'LDA.model'
    #lda_model = models.ldamodel.LdaModel.load(ldamodel_path)
    #doc=['mean','universe' ,'buffering' ,'us']
    #doc = ['Not ', 'really']
    '''
    for t in texts:
        doc = lda_model.id2word.doc2bow(t)
        #doc_topics, word_topics, phi_values = lda_model.get_document_topics(doc, per_word_topics=True)
        results = lda_model.get_document_topics(doc, per_word_topics=True)
        print results
        '''
    '''
    for i in range(1,num_topics):
        topic_models(corpus=corpus_tfidf, dictionary=dictionary,num_topics=i)
        lda_model = models.ldamodel.LdaModel.load(ldamodel_path)
        '''
        #test_perplexity(corpus_tfidf, i)
        #coherence =  CoherenceModel(model=lda_model, corpus=corpus_tfidf, texts=texts, dictionary=dictionary, coherence='u_mass').get_coherence()      
        #print CoherenceModel(model=lda_model, corpus=corpus_tfidf, texts=texts, dictionary=dictionary, coherence='u_mass').get_coherence()
        #print CoherenceModel(model=lda_model, corpus=corpus, texts=new_docs, dictionary=dictionary, coherence='c_uci').get_coherence()
        #print CoherenceModel(model=lda_model, corpus=corpus, texts=new_docs, dictionary=dictionary, coherence='c_npmi').get_coherence()
        #print coherence
    
       
    
          

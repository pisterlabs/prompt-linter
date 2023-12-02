import nltk
import Function
#nltk.download('wordnet')
#nltk.download('punkt')

from nltk.corpus import wordnet as wn
import gensim
from gensim import corpora,models
import pickle
import numpy as np
import random
import re    
import sys
import pandas as pd
from gensim.models import CoherenceModel
#reload(sys)
#sys.setdefaultencoding('utf8')

lineMapDict={} # to store Function object to tokens(from docstring) mapping

#takes doc as documentation,functionName
def tokenize(doc):
    tokens=nltk.word_tokenize(doc)
    tcopy=list(tokens)
    for t in tcopy:
        tokens.remove(t)
        tokens.extend(camel_case_split(t))
    tokens=[token.lower() for token in tokens]
    return tokens

"""Function to perform tokenization of a token such as perform camel case tokenization, split on hyphens,
   dots, underscores, slashes etc. and then return a new list of tokens obtained"""

def camel_case_split(identifier):
    identifier=identifier.replace('{','')
    identifier=identifier.replace('}','')
    
    x=identifier.split("-")
    #print x
    y=[]
    for j in x:
     z=j.split('_')
     y+=z
    x=y[:]
    y=[]
    for j in x:
     z=j.split("/")
     y+=z
    #print y
    x=[]
    for j in y:
     y1=j.split(".")
     x+=y1
    #print x
    y=[]
    for j in x:
     y1=j.split("=")
     y+=y1
    x=[]
    for j in y:
     y1=j.split("(")
     x+=y1
    matches=[]
    for j in y:
     y1=j.split("#")
     x+=y1
    for j in y:
     y1=j.split("<")
     x+=y1
    for j in y:
     y1=j.split(">")
     x+=y1
    for i in x:
     matches +=(re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', i))
    #print matches
    matches1=[]
    for i in matches:
     #print i.group(0)
     matches1.append(i.group(0).lower())
    #print matches1
    return matches1
    
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

#nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def readfromfile(fname,sheet):
    text_data = []
    df=pd.read_excel(fname,sheet_name=sheet,header=0)
    for a,b,c,d,e in zip(df['repo_name'], df['repo_path'], df['function sig'], df['method'], df['docstring']):
        func=Function.Function(a,b,c,d,e)
        line=str(e)
        tokens = prepare_text_for_lda(line)
        lineMapDict[func]=tokens
        text_data.append(tokens)
    return text_data


def get_function_explanation(docstring):
    rs = re.findall('(.*?)(Args|args|Usage|usage|Example|param|@return|rtype\:|return\:|type\:|throws|\Z)', docstring)[0][0]
    rs = drop_special_chars(drop_urls(drop_example(rs)))
    #print(rs)
    if rs.strip()=='':
        rs=docstring
    return rs


def remove_punctuations(sentence):
    return sentence.translate(translator).strip()


def drop_urls(sentence):
    return re.sub(r'http\S+', '', sentence)


def drop_example(sentence):
    return re.sub(r'(Example|Usage)(.*?)(param|return|\Z)', '', sentence)


def drop_special_chars(sentence):
    return re.sub(r"[^a-zA-Z0-9]+", ' ', sentence)


    
"""Perform preprocessing on the documentation text to tokenize and ready to feed to ldamodel
Preprocessing includes: stopwords elimination, splitting(on _ - camelcase etc.), lemmatization
"""    
def prepare_text_for_lda(text):
    #print(text)
    text=text.replace('\n',' ')#extract only function description for topic modeling
    text=get_function_explanation(text)#extract only function description for topic modeling
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 1]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma2(token) for token in tokens]
    return tokens

"""Prepare dictionary of words in the text to be used for modelling"""
def prep_dict(tokens,pname):
    tokens = np.asarray(tokens)
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]
    tfidf = models.TfidfModel(corpus)
    corpus = tfidf[corpus]
    pickle.dump(corpus, open(pname+'_corpus.pkl', 'wb'))
    dictionary.save(pname+'_dict.gensim')
    return (corpus,dictionary)

"""Extract topics from the corpus using LDA modelling 
and call groupDocs to group documents from the corpus to the closest topic"""
def extracttopics(fname,sheet,pname,NUM_TOPICS = 5):
    tokens= readfromfile(fname,sheet)# lda ready for of query tokens
    (corpus,dictionary)=prep_dict(tokens,pname)
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=40)
    ldamodel.save(pname+'_lda.gensim')
    topics = ldamodel.print_topics(num_words=5)
    #print(ldamodel.get_document_topics(corpus))     
    #for topic in topics:
    #    print(topic)
    MapDict=groupDocs(ldamodel, dictionary,tokens)
    with open(pname+'_cluster.txt', "wb") as myFile: # save grouping in a file to load later
        pickle.dump(MapDict, myFile)
        
    # Compute Perplexity
    print('\nPerplexity: ', ldamodel.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=tokens, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    return(ldamodel,dictionary,topics,MapDict)


"""Given a query as string or tokenized query,
returns the score and index of the top matched topics from the corpus"""
def gettopmatch(ldamodel,dictionary,qtokens=[],query=''): # optional to send query as text or as tokens
    if(len(qtokens)==0):
        qtokens=prepare_text_for_lda(query)
    #print(qtokens)
    new_doc_bow = dictionary.doc2bow(qtokens)
    #print(ldamodel.get_document_topics(new_doc_bow))
    doc_lda = ldamodel[new_doc_bow]   
    #print(sorted(doc_lda, key=lambda tup: -1*tup[1]))
    #print(ldamodel.print_topics(3, 5))
    #for index, score in sorted(doc_lda, key=lambda tup: -1*tup[1]):
    #    print(ldamodel.print_topic(index, 10))
    flag=False
    topgrp=[]
    for index, score in sorted(doc_lda, key=lambda tup: -1*tup[1]):
        if(not flag):
            top=score
            flag=True
        if(top-score<0.05):
            topgrp.append((score, index)) # return only the top match
    
    return topgrp 


"""Groups each document derived list of tokens to its closest topic. 
Groups sored as dictionary where key is the index of the topic from the Lda model
and value is list of list of tokens for close documentations"""    
def groupDocs(ldamodel, dictionary,tokens):
    MapDict={} # key as index (in lda) of closest topic,val as list of documentations close to the topic
    for q in tokens:
        topmatches=gettopmatch(ldamodel,dictionary,qtokens=q)
        for score,index in topmatches:
            if(index in MapDict):#val should be list of 'Function' objects
                MapDict[index].append(list(lineMapDict.keys())[list(lineMapDict.values()).index(q)])
            else:
                MapDict[index]=[]
                MapDict[index].append(list(lineMapDict.keys())[list(lineMapDict.values()).index(q)])
    return MapDict

def printClusters(pname):#print and write to excel with cluster id
    with open(pname+'_cluster.txt', "rb") as myFile:
        MapDict = pickle.load(myFile)
    ldamodel=models.LdaModel.load(pname+'_lda.gensim')
    #Enable to print topics
    """for k in MapDict:
        print("*********************")
        print(ldamodel.print_topic(k, topn=10))
        for v in MapDict[k]:
            print("+++++++++\n",v.doc)"""
    dclus=dict()
    dclus['ID']=[]
    dclus['CID']=[]
    dclus['c_topic']=[]
    dclus['repo_name']=[]
    dclus['repo_path']=[]
    dclus['function sig']=[]
    dclus['method']=[]
    dclus['docstring']=[]
    ctr=0
    for k in MapDict:
        for v in MapDict[k]:
            dclus['ID'].append(ctr)
            ctr=ctr+1
            dclus['CID'].append(k)
            dclus['c_topic'].append(str(ldamodel.print_topic(k, topn=10)))
            dclus['repo_name'].append(v.repo_name)
            dclus['repo_path'].append(v.file_path)
            dclus['method'].append(v.source)
            dclus['function sig'].append(v.sig)
            dclus['docstring'].append(v.doc)

	
    df=pd.DataFrame(dclus)# prepare by appeding to list of each col
    writer=pd.ExcelWriter(pname+'data_cluster.xlsx')
    df.to_excel(writer,'Sheet1',index=False)   
    writer.save()
    
if __name__ == "__main__":
    dataset='Functions_data2.xlsx'#TODO: change to larger xlsx
    sheet='utility'
    projectname='dataset_utility' # the one to use for training the lda model
    (ldamodel,dictionary,topics,MapDict)=extracttopics(dataset,sheet,projectname,120)
    printClusters(projectname)
        
    """
    q1="Split a path in root and extension. The extension is everything starting at the last dot in the last pathname component; the root is everything before that. It is always true that root + ext == p."
    q2="Returns the file extension for the given file name, or the empty string if the file has no extension. The result does not include the '.'. Note: This method simply returns everything after the last '.' in the file's name as determined by File.getName(). It does not account for any filesystem-specific behavior that the File API does not already account for. For example, on NTFS it will report \"txt\" as the extension for the filename \"foo.exe:.txt\" even though NTFS will drop the \":.txt\" part of the name when the file is actually created on the filesystem due to NTFS's Alternate Data Streams."
    q3="Parses a URL by using the default parsing options."
    q4="Given a url, return a parsed :class:`.Url` namedtuple. Best-effort is performed to parse incomplete urls. Fields not provided will be None. Partly backwards-compatible with :mod:`urlparse`."
    
    for q in [q1,q2,q3,q4]:
          gettopmatch(ldamodel,dictionary,query=q)
    """    

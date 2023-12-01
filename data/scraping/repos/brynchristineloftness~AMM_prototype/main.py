#!/usr/bin/python3

#import sys
import pandas as pd
import string
#import gensim
#from gensim.models import Word2Vec 
#from gensim.models import Doc2Vec
#from gensim.models import CoherenceModel
#from gensim.models import TfidfModel
#from gensim.corpora import Dictionary
import nltk
from nltk.tokenize import word_tokenize 
import xml.etree.ElementTree as ET
import re
#import collections
#from sklearn.decomposition import PCA
#from matplotlib import pyplot as plt
#from string import digits
#from gensim import corpora
#import pprint
#from gensim import models
#from gensim import similarities 
#from matplotlib import colors
#import numpy as np
#from collections import OrderedDict
#from xml.etree.ElementTree import XML, fromstring, tostring
#from xml.etree.ElementTree import Element, SubElement, Comment
#import lxml
#import keras
#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
#import tensorflow as tf
#from keras.preprocessing.text import Tokenizer
#import io
#from xml.etree.ElementTree import XML, fromstring
#import itertools
#from functools import reduce
#from IPython.display import display, HTML


def sortstuff(name):
    name = sorted(name)
    for item in range(len(name)):
        name[item] = sorted(name[item])
    name = sorted(name)
    name_set = set(tuple(x) for x in name)
    name = [ list(x) for x in name_set ]
    return name

def defineoracle_optionbuilder():
    oracle = [['testCompleteOption','test02'],['testCompleteOption','test05'],['testCompleteOption','test08'],
          ['testCompleteOption','test19'],['testCompleteOption','test22'], ['testTwoCompleteOptions','test08'],
          ['testTwoCompleteOptions','test19'],['testTwoCompleteOptions','test22'], ['testBaseOptionCharOpt','test08'],
          ['testIllegalOptions', 'test14'],['testSpecialOptChars','test15'],['testCreateIncompleteOption','test16'],
          ['testOptionArgNumbers','test21'],['testCompleteOption','testTwoCompleteOptions'],['testCompleteOption','testBaseOptionCharOpt'],
         ['testBaseOptionCharOpt','testTwoCompleteOptions'],['test02','test05'],['test02','test08'],['test02','test19'],
          ['test02','test22'],['test05','test08'],['test05','test19'],['test05','test22'],['test08','test19'],
          ['test08','test22']]     
    #merges and partial merge oracle       
    mpmoracle = [['testCompleteOption','test02'],['testCompleteOption','test05'],['testCompleteOption','test08'],
          ['testCompleteOption','test19'],['testCompleteOption','test22'], ['testTwoCompleteOptions','test08'],
          ['testTwoCompleteOptions','test19'],['testTwoCompleteOptions','test22'], ['testBaseOptionCharOpt','test08'],
          ['testIllegalOptions', 'test14'],['testSpecialOptChars','test15'],['testCreateIncompleteOption','test16'],
          ['testOptionArgNumbers','test21'], ['testCompleteOption','test06'], ['testCompleteOption','test28'], 
              ['testCompleteOption','test29'],['testTwoCompleteOptions','test28'],['testTwoCompleteOptions','test29'],
              ['testCompleteOption','testTwoCompleteOptions'],['testCompleteOption','testBaseOptionCharOpt'],
         ['testBaseOptionCharOpt','testTwoCompleteOptions'],['test02','test05'],['test02','test08'],['test02','test19'],
          ['test02','test22'],['test05','test08'],['test05','test19'],['test05','test22'],
              ['test08','test19'],['test08','test22'],['test02','test06'],['test02','test28'],['test02','test29'],
              ['test05','test06'],['test05','test28'],['test05','test29'],['test08','test06'],['test08','test28'],['test08','test29'],
              ['test19','test06'],['test19','test28'],['test18','test29'],['test22','test06'],['test22','test28'],['test22','test29']]          
    #sort and get rid of duplicates
    oracle = sortstuff(oracle)
    mpmoracle = sortstuff(mpmoracle)  
    oraclecluster = []        
    for file in oracle:
        oraclecluster.append(file[0])
        oraclecluster.append(file[1])

    oraclecluster = list(dict.fromkeys(oraclecluster))
    mpmoraclecluster = []        
    for file in mpmoracle:
            mpmoraclecluster.append(file[0])
            mpmoraclecluster.append(file[1])
    mpmoraclecluster = list(dict.fromkeys(mpmoraclecluster))
    return oracle, mpmoracle, oraclecluster,mpmoraclecluster

def definetestnames(myfile):     
    x = []
    testlist = myfile['Test'].tolist()
    x = re.compile('test')
    listofnames = []
    for item in testlist: 
        txt = str(item)
        x = re.findall(r"\btest\w+", txt)
        listofnames.append(x)
    myfile['TestName'] = listofnames
    myfile['TestName'] = myfile['TestName'].apply(lambda x:''.join([i for i in x if i not in string.punctuation])) 
    return myfile

def cleaning(myfile):
    myfile = myfile.replace(r'\n',' ', regex=True)    
    myfile['Scenario'] = myfile['Scenario'].apply(
        lambda x:''.join([i for i in x if i not in string.punctuation])) 
    myfile['Test'] = myfile['Test'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
    return myfile

def camel_case_split(str):
    words = [[str[0]]] 
    for c in str[1:]: 
        if words[-1][-1].islower() and c.isupper(): 
            words.append(list(c)) 
        else: 
            words[-1].append(c) 
    return [''.join(word) for word in words]
    
def camelcasing(myfile):
    myfile['Scenario'] = myfile['Scenario'].apply(lambda x:[i for i in camel_case_split(x)])
    myfile['Scenario'] = myfile['Scenario'].apply(lambda x:' '.join([i for i in x]))
    myfile['Test'] = myfile['Test'].apply(lambda x:[i for i in camel_case_split(x)])
    myfile['Test'] = myfile['Test'].apply(lambda x:' '.join([i for i in x]))
    return myfile

def lowercasing_and_backup(myfile):
    myfile['Test'] = myfile['Test'].apply(lambda x:''.join([i for i in x.lower()])) 
    myfile['Scenario'] = myfile['Scenario'].apply(lambda x:''.join([i for i in x.lower()])) 
    myfile['Combo'] = myfile['Scenario'].str.cat(myfile['Test'],sep = " ")
    myfile.columns = ['Type','Scenario','Test','TestName','Combo']
    mybackupfile = myfile.copy()
    return myfile

def tokenize_and_stopwords(myfile):
    myfile['Scenario'] = myfile.apply(lambda column: nltk.word_tokenize(column['Scenario']),axis = 1)
    myfile['Test'] = myfile.apply(lambda column: nltk.word_tokenize(column['Test']),axis = 1)
    myfile['Combo'] = myfile.apply(lambda column: nltk.word_tokenize(column['Combo']),axis = 1)
    stopwords=['a','an','and','is','of','its','it']
    myfile['Combo']= myfile['Combo'].apply(lambda x: [item for item in x if item not in stopwords])
    return myfile

def producelist(myfile):
    scenariocorp = myfile['Scenario'].tolist()
    testcorpus = myfile['Test'].tolist()
    combinedcorpus = myfile['Combo'].tolist()   
    return scenariocorp,testcorpus,combinedcorpus

def textpreprocessing(myfile):
    myfile = cleaning(myfile)
    myfile = definetestnames(myfile)
    myfile = camelcasing(myfile)
    myfile = lowercasing_and_backup(myfile)
    myfile = tokenize_and_stopwords(myfile)
    scenariocorpus,testcorpus,combinedcorpus = producelist(myfile)
    testlen = len(myfile['Test'])
    return myfile, testlen,scenariocorpus,testcorpus,combinedcorpus

def build_XML_trees():
    #removed all leading comments before declaration of package in original java files
    #used srcml to produce xml files for new clean java test files
    cleanautotree = ET.parse(r'cleanautotests.xml')
    cleanautoroot = cleanautotree.getroot()
    cleanmanualtree = ET.parse(r'cleanmanualtests.xml')
    cleanmanualroot = cleanmanualtree.getroot() 
    return cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot 

def cleantrees(cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot):
    for child in cleanautoroot:
        child.tag = child.tag.replace('{http://www.srcML.org/srcML/src}','')
        child.tag = child.tag.replace('class','startclass')
    for child in cleanmanualroot:
        child.tag = child.tag.replace('{http://www.srcML.org/srcML/src}','')
        child.tag = child.tag.replace('class','startclass')
    for tags in cleanautoroot.iter():
        tags.tag = tags.tag.replace('{http://www.srcML.org/srcML/src}','')
    for tags in cleanmanualroot.iter():
        tags.tag = tags.tag.replace('{http://www.srcML.org/srcML/src}','')
    return cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot

def processingXML(cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot,myfile):
    listofmanualfiles = []
    startclassElement = cleanmanualroot.find('startclass/block')
    for element in startclassElement: #isolates comment, function, comment, function for all 
        if (element.tag != 'comment'): #isolates only function blocks
            listofmanualfiles.append(ET.tostring(element, encoding='unicode')) 
    listofautofiles = []
    startclassElement = cleanautoroot.find('startclass/block')
    for element in startclassElement: #isolates comment, function, comment, function for all 
        if (element.tag != 'comment'): #isolates only function blocks
            listofautofiles.append(ET.tostring(element, encoding='unicode')) 
    listofallfiles = []
    listofallfiles = listofmanualfiles + listofautofiles
    myfile['XML'] = ['' for x in range(len(myfile['Combo']))]
    myfile.columns = ['Type','Scenario','Test','TestName','Combo','XML']
    myfile['XML'] = listofallfiles
    myfile['XML'] = [entry.replace('4000','') for entry in myfile['XML']]
    for file in range(len(myfile['XML'])):
        myfile['XML'][file] = '<root>' + myfile['XML'][file] + '</root>'
    for file in range(len(listofmanualfiles)):
        listofmanualfiles[file] = '<root>' + listofmanualfiles[file] + '</root>'
    for file in range(len(listofautofiles)):
        listofautofiles[file] = '<root>' + listofautofiles[file] + '</root>'
    for file in range(len(listofallfiles)):
        listofallfiles[file] = '<root>' + listofallfiles[file] + '</root>'
    myfile['XML'] = [entry.replace(r'\n','') for entry in myfile['XML']]
    myfile['Methods'] = ['' for x in range(len(myfile['Combo']))]
    myfile.columns = ['Type','Scenario','Test','TestName','Combo','XML','Methods']
    myfile['Methods'] = listofallfiles
    myfile['Asserts'] = ['' for x in range(len(myfile['Combo']))]
    myfile.columns = ['Type','Scenario','Test','TestName','Combo','XML','Methods', 'Asserts']
    myfile['Asserts'] = listofallfiles
    myfile['Methods_Asserts'] = ['' for x in range(len(myfile['Combo']))]
    myfile.columns = ['Type','Scenario','Test','TestName','Combo','XML','Methods', 'Asserts','Methods_Asserts']
    myfile['Methods_Asserts']=listofallfiles
    myfile['Assert_Only'] = ['' for x in range(len(myfile['Combo']))]
    myfile.columns = ['Type','Scenario','Test','TestName','Combo','XML','Methods', 'Asserts','Methods_Asserts','Assert_Only']
    myfile['Assert_Only']=listofallfiles
    myfile.columns = ['Type','Scenario','Test','TestName','Combo','XML','Methods', 'Asserts','Methods_Asserts','Assert_Only']
    return myfile,listofallfiles

def info_extraction_XML(myfile,listofallfiles):
    #isolating methods and assertions
    counter = 0
    for file in listofallfiles:
        #myfile['Methods'][counter] = ''
        #myfile['Asserts'][counter] = ''
        #myfile['Methods_Asserts'][counter] = ''
        stringstuff = ''
        assertstuff = ''
        expressionstuff = ''
        finalassertname = ''
        namestring = ''
        rootspot = ET.fromstring(file)
        findfunction = rootspot.find('function/block/block_content')
        tagstring = (ET.tostring(findfunction,encoding = 'unicode'))
        for element in findfunction:
            if (element.tag == 'decl_stmt'):
                for el in element:
                    if el.tag == 'decl':
                        for item in el:
                            if item.tag == 'init':
                                for call in item:
                                    if call.tag == 'expr':
                                        for name in item:
                                            if name.tag == 'expr':
                                                for item in name:
                                                    for call in item:
                                                        if call.tag == 'name':
                                                            methods_notrycatch_noasserts = (ET.tostring(call,encoding = 'unicode'))
                                                            stringstuff+=methods_notrycatch_noasserts
                                                            expressionstuff +=methods_notrycatch_noasserts
            elif(element.tag=='try'):
                tagstring = (ET.tostring(element,encoding = 'unicode'))
                for item in element:
                    if item.tag=='block':
                        for block in item:
                            if block.tag =='block_content':
                                for trystmt in block:
                                    if trystmt.tag == 'expr_stmt':
                                        for element in trystmt:
                                            if element.tag=='expr':
                                                for expr in element:
                                                    if expr.tag =='call':
                                                        for child in expr:
                                                            if child.tag == 'name':
                                                                trystuff = (ET.tostring(child,encoding = 'unicode'))
                                                                stringstuff += trystuff
                                                                expressionstuff += trystuff
            elif(element.tag == 'expr_stmt'):
                for element2 in element:
                    if element2.tag == 'expr':
                        for expr in element2:
                            if expr.tag == 'call':
                                for call in element:
                                    if call.tag == 'expr':
                                        for expr in call:
                                            if expr.tag =='call':
                                                for file in expr:
                                                    if file.tag == 'name':
                                                        assertname = (ET.tostring(file,encoding = 'unicode'))
                                                        assertstuff += assertname
                                                        expressionstuff += assertname
                                                        finalassertname = assertname
                                                    elif file.tag =='argument_list':
                                                        for element in file:
                                                            if element.tag == 'argument':
                                                                for item in element:
                                                                    if item.tag == 'expr':
                                                                        for child in item:
                                                                            if child.tag == 'call':
                                                                                for children in child:
                                                                                    if children.tag == 'name':
                                                                                        for name in children:
                                                                                            if name.tag =='name':
                                                                                                namespot = (ET.tostring(name,encoding = 'unicode'))
                                                                                                assertstuff += namespot
                                                                                                expressionstuff+=namespot

        myfile['Methods'][counter] = stringstuff
        myfile['Asserts'][counter] = assertstuff
        myfile['Methods_Asserts'][counter] = expressionstuff
        myfile['Assert_Only'][counter] = finalassertname
        counter += 1
    clean('Methods',myfile)
    clean('Asserts',myfile)
    clean('Methods_Asserts',myfile)
    clean('Assert_Only',myfile) 
    for file in range(len(myfile['Asserts'])):
        myfile['Asserts'][file] = myfile['Asserts'][file].split()
    for file in range(len(myfile['Methods'])):
        myfile['Methods'][file] = myfile['Methods'][file].split()
    for file in range(len(myfile['Methods_Asserts'])):
        myfile['Methods_Asserts'][file] = myfile['Methods_Asserts'][file].split()
    for file in range(len(myfile['Assert_Only'])):
        myfile['Assert_Only'][file] = myfile['Assert_Only'][file].split()
    return myfile

def clean(column,myfile):
    myfile[column] = [entry.replace('/','') for entry in myfile[column]]      
    myfile[column] = [entry.replace('<call>',' ') for entry in myfile[column]]
    myfile[column] = [entry.replace('<operator>',' ') for entry in myfile[column]]
    myfile[column] = [entry.replace('<name>',' ') for entry in myfile[column]]
    myfile[column] = [entry.replace('<argument>',' ') for entry in myfile[column]]
    myfile[column] = [entry.replace('<list>',' ') for entry in myfile[column]]
    myfile[column] = [entry.replace('<argument_list>',' ') for entry in myfile[column]]
    myfile[column] = [entry.replace('<expr>',' ') for entry in myfile[column]]
    myfile[column] = [entry.replace('<literal>',' ') for entry in myfile[column]]
    myfile[column] = [entry.replace('<char>',' ') for entry in myfile[column]]
    myfile[column] = [entry.replace('<type>',' ') for entry in myfile[column]]
    myfile[column] = myfile[column].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
    return myfile

def one2one_methods_plus_assertchecking_results(myfile,testlen):
    testlist_set = myfile['Methods']
    matchlist = []
    for test in range(testlen):
        for test2 in range(testlen):
            if test !=test2:
                testone = testlist_set[test]
                testtwo = testlist_set[test2]
                if testone == testtwo:#if the methods are the same
                    if len(myfile['Asserts'][test]) == 0 or len(myfile['Asserts'][test2]) == 0 or myfile['Asserts'][test][-1] == myfile['Asserts'][test2][-1]:
                        matchlist.append([myfile['TestName'][test],myfile['TestName'][test2]])                   
    matchlist = sortstuff(matchlist)
    pack2 = matchlist 
    return pack2

def one2one_asserts_results(myfile,testlen):
    testlist_set = myfile['Asserts']
    matchlist = []
    for test in range(testlen):
        for test2 in range(testlen):
            if test !=test2:
                testone = testlist_set[test]
                testtwo = testlist_set[test2]
                if testone == testtwo and testone == []:
                    matchlist.append([myfile['TestName'][test],myfile['TestName'][test2]])
    matchlist = sortstuff(matchlist)
    pack3 = matchlist #pack3
    return pack3

def lcs(X, Y): 
    m = len(X) 
    n = len(Y) 
    L = [[None]*(n + 1) for i in range(m + 1)] 
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 

def longestcommonsubsequence(myfile,testlen):
    testlist_set = myfile['Asserts']
    sublist = []
    sublist2 = []
    for test in range(testlen):
        for test2 in range(testlen):
            if test !=test2:
                testone = testlist_set[test]
                testtwo = testlist_set[test2]
                subsequencelen = lcs(testone,testtwo)
                minimum = min(len(testone),len(testtwo))
                subsequencelen = lcs(testone,testtwo)
                min_minus_lcs = minimum-subsequencelen
                if subsequencelen>4:
                    sublist.append([myfile['TestName'][test],myfile['TestName'][test2]])
                elif subsequencelen >2 : sublist2.append([myfile['TestName'][test],myfile['TestName'][test2]])
    sublist = sortstuff(sublist)
    sublist2 = sortstuff(sublist2)
    pack9 = sublist 
    pack10 = sublist2 
    return pack9, pack10

def createsortedlist(grid):
    main_list = []
    counter = 0
    official_list = []
    stuff = ['','','']
    official_list.append(stuff)
    for i,row in enumerate(grid):
        for j, cell in enumerate(row):
            if (i!=j):
                stuff = [i, j, grid[i][j]]
                main_list.append(stuff)
    mainlistsorted = sortstuff(main_list)
    mainlistsorted = sorted(main_list, key=lambda x: x[-1], reverse=True)
    return official_list, mainlistsorted

def compute(officiallist, sortedlist, breakpoint):
    valuelist = []
    for group in sortedlist:
        if(group[2] > breakpoint):
            if (group[0]!=officiallist[-1][1] and group[1] != officiallist[-1][0]):
                officiallist.append(group)
    officiallist.remove(officiallist[0])
    return officiallist,sortedlist

def printresults(officiallist,stringstuff):
    full_list = []
    index_list = []
    info = ''
    print(stringstuff)   
    for thing in officiallist:
        info = [myfile['TestName'][thing[0]], myfile['TestName'][thing[1]]]
        indexinfo = [thing[0], thing[1]]
        index_list.append(indexinfo)
        full_list.append(info)
    results = sortstuff(full_list)
    index_list = sortstuff(index_list)
    return results, index_list

def full_output_of_lists(full_list,oracle,mpmoracle,truepositivelist,falsepositivelist,truepositivelistmpm,falsepositivelistmpm,testcluster,truepositive,falsepositive,tporacle,fporacle):
    print('true positive list:')
    for file in truepositivelist:
        print(file)
    print()
    print("List of missed positives(false negatives) for merge-only oracle")
    for file in oracle:
        if file not in truepositivelist:
            print(file)
    print()   
    print("List of true positives for merge-only oracle")
    for file in oracle:
        if file in truepositivelist:
            print(file)
    print()        
    print("List of false positives for merge-only oracle")
    for file in falsepositivelist:
        print(file)
    print()        
    print("List of missed positives(false negatives) for merge/partial-merge oracle")
    for file in mpmoracle:
        if file not in truepositivelistmpm:
            print(file)
    print()
    print("List of true positives for merge/partial-merge oracle")
    for file in mpmoracle:
        if file in truepositivelistmpm:
            print(file) 
    print()
    print("List of false positives for merge/partial-merge oracle")
    for file in falsepositivelistmpm:
        print(file)

def output(full_list,oracle,mpmoracle,truepositivelist,falsepositivelist,truepositivelistmpm,falsepositivelistmpm,testcluster,truepositive,falsepositive,tporacle,fporacle):
    print(len(testcluster),testlen)
    print('total in MPMoracle;',len(mpmoracle),'total predictions',len(full_list))
    print('true pos:',truepositive, '  ', 'false pos', falsepositive)
    if len(full_list) !=0:
        tprate = truepositive/len(full_list)
        fprate = falsepositive/len(full_list)
    else :
        tprate = 0
        fprate = 0
    #print("-----------")
    print('total in oracle',len(oracle),'total predictions',len(full_list))
    print('true pos:',tporacle, '  ', 'false pos', fporacle)
    if len(full_list) !=0:
        tprate_o = tporacle/len(full_list)
        fprate_o = fporacle/len(full_list)
    else :
        tprate_o = 0
        fprate_o = 0
    print('-----------')
    print('ORACLE:')
    print('tp rate:', tprate_o, '  ', 'fp rate:', fprate_o)
    #print(tprate_o, ',', fprate_o,',', tprate, ',',fprate)
    print('MPM ORACLE:')
    print('tp rate:', tprate, '  ', 'fp rate:', fprate)
    full_output_of_lists(full_list,oracle,mpmoracle,truepositivelist,falsepositivelist,truepositivelistmpm,falsepositivelistmpm,testcluster,truepositive,falsepositive,tporacle,fporacle)


def TPFPoutput(full_list,oracle,mpmoracle):
    testcluster = []
    full_list = sorted(full_list)
    fl_set = set(tuple(x) for x in full_list)
    full_list = [ list(x) for x in fl_set ]
    for item in full_list:
        item = sorted(item)
    truepositive = 0
    falsepositive = 0
    tporacle = 0
    fporacle = 0
    truepositivelist = []
    falsepositivelist= []
    truepositivelistmpm= []
    falsepositivelistmpm = []
    for file in full_list:
        testcluster.append(file[0])
        testcluster.append(file[1])
        if file in mpmoracle:
            truepositive+=1
            truepositivelistmpm.append(file)
        else:
            falsepositive+=1
            falsepositivelistmpm.append(file)
        if file in oracle:
            tporacle += 1
            truepositivelist.append(file)
        else:
            fporacle +=1   
            falsepositivelist.append(file)
    testcluster = list(dict.fromkeys(testcluster))
    truepositivelist = sortstuff(truepositivelist)
    falsepositivelist = sortstuff(falsepositivelist)
    truepositivelistmpm = sortstuff(truepositivelistmpm)
    falsepositivelistmpm = sortstuff(falsepositivelistmpm)
    output(full_list,oracle,mpmoracle,truepositivelist,falsepositivelist,truepositivelistmpm,falsepositivelistmpm,testcluster,truepositive,falsepositive,tporacle,fporacle)
    return full_list

def intersect(name):
    setcountertest = 0
    setcountertest2 = 0
    for test in range(testlen):
        for item in set((myfile[name][test])):
                setcountertest += 1
        for test2 in range(testlen):
            for item in set((myfile[name][test2])):
                setcountertest2 += 1
            intersection = set((myfile[name][test])).intersection(set((myfile[name][test2])))
            intersection = len(intersection)
            minimum = min(setcountertest,setcountertest2)
            if minimum != 0:
                metric = intersection/minimum
            else:
                metric = 0
            #print(setcountertest,setcountertest2,test,test2,intersection,metric)
            intersectiongrid[test][test2] = metric
            if (test == test2):
                intersectiongrid[test][test2] = 0
            setcountertest2 = 0
            intersection = 0
        setcountertest = 0
    return intersectiongrid

def setmetrics_combo(myfile):
    intersectiongrid = defaultgrid     
    intersectiongrid = intersect('Combo')
    official_list = []
    listsorted = []
    official_list, listsorted = createsortedlist(intersectiongrid)
    listsorted = sorted(listsorted, key=lambda x: x[0])
    listsorted = list(set(tuple(x) for x in listsorted))
    official_list, listsorted = compute(official_list, listsorted,.51)
    setintersectionresults, index_list_methods = printresults(official_list,'Full Results for Methods (no args or suite name)')
    results = TPFPoutput(setintersectionresults,oracle,mpmoracle)
    pack15 = index_list_methods 
    return pack15

def intersectwithasserts(name):
    setcountertest = 0
    setcountertest2 = 0
    for test in range(testlen):
        for item in set((myfile[name][test])):
                setcountertest += 1
        for test2 in range(testlen):
            for item in set((myfile[name][test2])):
                setcountertest2 += 1
            intersection = set((myfile[name][test])).intersection(set((myfile[name][test2])))
            intersection = len(intersection)
            minimum = min(setcountertest,setcountertest2)
            if minimum != 0:
                metric = intersection/minimum
            else:
                metric = 0
            if 'assertNotNull' in myfile[name][test] and 'assertEquals' in myfile[name][test2]:
                metric += 1
            #print(setcountertest,setcountertest2,test,test2,intersection,metric)
            intersectiongrid[test][test2] = metric
            if (test == test2):
                intersectiongrid[test][test2] = 0
            setcountertest2 = 0
            intersection = 0
        setcountertest = 0
    return intersectiongrid

def intersect_withassertsadditive_results(myfile):
    assertgrid = defaultgrid
    assertgrid = intersectwithasserts('Asserts')
    officiallist, listsorted = createsortedlist(assertgrid)
    for item in listsorted:
        if [item[1],item[0],item[2]] in listsorted:
            listsorted.remove(item)  
    listsorted = sorted(listsorted, key=lambda x: x[0])
    listsorted = list(set(tuple(x) for x in listsorted))
    officiallist, listsorted = compute(officiallist, listsorted, .99)
    setintersectionresults, index_list = printresults(officiallist,'Full Results for Asserts')
    results = TPFPoutput(setintersectionresults,oracle,mpmoracle)
    pack21 =setintersectionresults 
    return pack21

def computelower(officiallist, sortedlist, breakpoint):
    valuelist = []
    print('breakpoint: ', breakpoint)
    for group in sortedlist:
        if(group[2] < breakpoint):
            if (group[0]!=officiallist[-1][1] and group[1] != officiallist[-1][0]):
                officiallist.append(group)
    officiallist.remove(officiallist[0])
    return officiallist,sortedlist

def scenarioskipgram(scenariocorpus,myfile,testlen):
    scenariomodel_skipgram = Word2Vec(scenariocorpus,window=5,min_count=1,iter = 15,alpha=.2,sg=1,size = 75,seed = 0)
    scenariomodelskipgram_grid = defaultgrid
    for test in range(testlen):
        for test2 in range(testlen):
            num = scenariomodel_skipgram.wv.n_similarity(myfile['Scenario'][test],myfile['Scenario'][test2])
            scenariomodelskipgram_grid[test][test2] = num
    minimum = 1.0
    for i,row in enumerate(scenariomodelskipgram_grid):
        for j, cell in enumerate(row):
            if (cell < minimum):
                minimum = scenariomodelskipgram_grid[i][j] 
    official_list_skipgram, mainlistsorted = createsortedlist(scenariomodelskipgram_grid)
    mainlistsorted = sorted(mainlistsorted, key=lambda x: x[0])
    mainlistsorted = list(set(tuple(x) for x in mainlistsorted))
    official_list_skipgram, mainlistsorted = computelower(official_list_skipgram, mainlistsorted,.82)
    skipgramresults, index_list = printresults(official_list_skipgram,'Full Results for Skipgram')
    results = TPFPoutput(skipgramresults,oracle,mpmoracle)
    prune1 = skipgramresults
    return prune1

def tfidfcorptogrid(entries,testlen,tfidfgrid,IR):
    entries = [[ele for ele in sub if not ele.isdigit()] for sub in entries] 
    dict_for_tfidf = Dictionary(entries)
    corp = [dict_for_tfidf.doc2bow(line) for line in entries]
    tfidfmodel = TfidfModel(corp,smartirs = IR)   
    corp_tfidf = tfidfmodel[corp]
    index_tfidf = similarities.MatrixSimilarity(corp_tfidf)
    sims= index_tfidf[corp_tfidf]
    for i,s in enumerate(sims):
        for counter in range(testlen):
            tfidfgrid[i][counter] = s[counter]
            if (i == counter):
                tfidfgrid[i][counter] = 0
    return tfidfgrid

def tfidf_nfc(scenariocorpus,testlen,myfile):
    tfidfgrid = defaultgrid
    tfidfgrid = tfidfcorptogrid(scenariocorpus,testlen,tfidfgrid,'nfc') 
    official_list_tfidf, mainlistsorted = createsortedlist(tfidfgrid)
    for item in mainlistsorted:
        if [item[1],item[0],item[2]] in mainlistsorted:
            mainlistsorted.remove(item)
    mainlistsorted = sorted(mainlistsorted, key=lambda x: x[0])
    mainlistsorted = list(set(tuple(x) for x in mainlistsorted))
    official_list_tfidf, mainlistsorted = compute(official_list_tfidf, mainlistsorted,.5)
    tfidfresults, index_list = printresults(official_list_tfidf,'Full Results for TFIDF')
    results = TPFPoutput(tfidfresults,oracle,mpmoracle)
    pack23 = tfidfresults
    return pack23

def tfidf_bnn(scenariocorpus,testlen,myfile):
    tfidfgrid = tfidfcorptogrid(scenariocorpus,testlen,tfidfgrid,'bnn')
    official_list_tfidf, mainlistsorted = createsortedlist(tfidfgrid)
    mainlistsorted = sorted(mainlistsorted, key=lambda x: x[0])
    mainlistsorted = list(set(tuple(x) for x in mainlistsorted))
    official_list_tfidf, mainlistsorted = compute(official_list_tfidf, mainlistsorted,.85)
    tfidfresults, index_list = printresults(official_list_tfidf,'Full Results for TFIDF')
    results = TPFPoutput(tfidfresults,oracle,mpmoracle)
    pack24 = tfidfresults
    return pack24

def LSI(myfile):
    entries = myfile['Scenario'].tolist()
    entries = [[ele for ele in sub if not ele.isdigit()] for sub in entries] 
    dict_for_lsi = Dictionary(entries)
    corp = [dict_for_lsi.doc2bow(line) for line in entries]
    lsi = models.LsiModel(corp,num_topics = 3)
    corp_lsi = lsi[corp]
    index_lsi = similarities.MatrixSimilarity(corp_lsi) 
    sims= index_lsi[corp_lsi]
    lsi_grid = defaultgrid
    for i,s in enumerate(sims):
        for counter in range(testlen):
            lsi_grid[i][counter] = s[counter]
            if (i == counter):
                lsi_grid[i][counter] = 0     
    official_lsi_list, lsi_listsorted = createsortedlist(lsi_grid)
    listsorted = sorted(listsorted, key=lambda x: x[0])
    listsorted = list(set(tuple(x) for x in listsorted))
    official_lsi_list, lsi_listsorted = computelower(official_lsi_list, lsi_listsorted,.70)
    lsiresults, index_list = printresults(official_lsi_list,'Full Results for LSI')
    results = TPFPoutput(lsiresults,oracle,mpmoracle)
    prune4 = lsiresults 
    return prune4

def create_suitetype_lists(myfile):
    manuallist = []
    autolist = []
    for item in range(len(myfile['Type'])):
        if myfile['Type'][item] == "Manual":
            manuallist.append(myfile['TestName'][item])
        else:
            autolist.append(myfile['TestName'][item])
    return manuallist, autolist


def round1_computation(autolist,manuallist,pack24):
    round1 = []
    round1 = pack24
    round1 = sortstuff(round1)
    for item in round1:
        if item[0] in manuallist and item[1] in autolist:
            round1.remove(item)
        elif item[0] in autolist and item[1] in manuallist:
            round1.remove(item)
    print('round1',len(round1)) 
    return round1 

def round2_computation(autolist,manuallist,pack3):
    round2 = []
    round2 = pack3
    round2 = [x for x in round2 if x not in round1]
    round2 = sortstuff(round2)
    for item in round2:
        if item[0] in manuallist and item[1] in manuallist:
            round2.remove(item)
        elif item[0] in autolist and item[1] in autolist:
            round2.remove(item)
    print('round2',len(round2),2)
    return round2

def round3_computation(autolist,manuallist,pack9):
    round3 = []
    round3 = (pack9)
    round3 = [x for x in round3 if x not in round1]
    round3 = [x for x in round3 if x not in round2]
    round3 = sortstuff(round3)
    for item in round3:
        if item[0] in manuallist and item[1] in autolist:
            round3.remove(item)
        elif item[0] in autolist and item[1] in autolist:
            round3.remove(item)
    print('round3',len(round3),3)
    return round3

def round4_computation(autolist,manuallist,pack10):
    round4 = []
    round4 = pack10
    round4 = [x for x in round4 if x not in round1]
    round4 = [x for x in round4 if x not in round2]
    round4 = [x for x in round4 if x not in round3]
    round4 = sortstuff(round4)
    for item in round4:
        if item[0] in manuallist and item[1] in manuallist:
            round4.remove(item)
        elif item[0] in autolist and item[1] in autolist:
            round4.remove(item)
    print('round4',len(round4),2)
    return round4

def round5_computation(autolist,manuallist,pack23):
    round5 = []
    round5 = pack23
    round5 = [x for x in round5 if x not in round1]
    round5 = [x for x in round5 if x not in round2]
    round5 = [x for x in round5 if x not in round3]
    round5 = [x for x in round5 if x not in round4]
    round5 = sortstuff(round5)
    for item in round5:
        if item[0] in manuallist and item[1] in manuallist:
            round5.remove(item)
        elif item[0] in autolist and item[1] in autolist:
            round5.remove(item)
    print('round5',len(round5),1)
    return round5

def round6_computation(autolist,manuallist,pack21):
    round6 = pack21 
    round6 = [x for x in round6 if x not in round1]
    round6 = [x for x in round6 if x not in round2]
    round6 = [x for x in round6 if x not in round3]
    round6 = [x for x in round6 if x not in round4]
    round6 = [x for x in round6 if x not in round5]
    round6 = sortstuff(round6)
    round6real = []
    for item in round6:
        if item[0] in autolist and item[1] in autolist:
            round6.remove(item)
        elif item[0] in manuallist and item[1] in manuallist:
            round6.remove(item)
        else: round6real.append(item)
    round6 = round6real
    print('round6',len(round6),4)
    return round6

def round7_computation(autolist,manuallist,pack15):
    round7 = pack15
    round7 = [x for x in round7 if x not in round1]
    round7 = [x for x in round7 if x not in round2]
    round7 = [x for x in round7 if x not in round3]
    round7 = [x for x in round7 if x not in round4]
    round7 = [x for x in round7 if x not in round5]
    round7 = [x for x in round7 if x not in round6]
    round7 = sortstuff(round7)
    round7real = []
    for item in round7:
        if item[0] in manuallist and item[1] in autolist:
            round7.remove(item)
        elif item[0] in autolist and item[1] in manuallist:
            round7real.append(item)
    round7 = round7real
    print('round7',len(round7),6)
    return round7

def definemissingvalues(finallist,oracle,mpmoracle):
    print()
    print('Missing 1 to 1 pairs:') 
    for pair in oracle:
        if pair not in finallist:
            print('MISSING in ORACLE one to one',pair)         
    for pair in mpmoracle:
        if pair not in finallist:
            print('MISSING in mpmORACLE one to one',pair)   
    print()
    testcluster = []        
    for file in finallist:
            testcluster.append(file[0])
            testcluster.append(file[1])
    testcluster = list(dict.fromkeys(testcluster))
    
    print('testcluster',testcluster)
    print()
    print('oraclecluster',oraclecluster)
    
    print('len of oracle cluster list', len(oraclecluster))
    print('len of mpmoracle cluster list', len(mpmoraclecluster))
    print('len of test cluster list',len(testcluster))
    print()
    print('Missing tests:')
    for file in oraclecluster:
        if file not in testcluster:
            print("MISSINGo", file) 
        else: print('in_o',file)
    for file in mpmoraclecluster:
        if file not in testcluster:
            print("MISSINGmpm", file)
        else: print('in_mpm',file)


def prototypecheck(oracle,mpmoracle,unit):
    pack1 = sortstuff(unit)
    counter = 0
    counter2 = 0
    itemlist = []
    for item in unit:  
        if item in oracle:
            itemlist.append(item)
            counter +=1
        else: 
            counter2+=1
    print("Number of found combos", counter)
    print('printing items found in matchlist that are in oracle:')
    for item in sorted(itemlist):
        print(item) 
    print()
    for item in unit:  
        if item in mpmoracle:
            itemlist.append(item)
            counter +=1
        else: 
            counter2+=1
    print("Number of found combos", counter)
    print('printing items found in matchlist that are in mpmoracle:')
    for item in sorted(itemlist):
        print(item) 
    print()
    question1 = sorted(unit)[0][0]
    questionnum =1
    for item in sorted(unit):
        print(item)
        if question1 != item[0]:
            question1 = item[0]
            questionnum+=1
    print('number of questions', questionnum)
    definemissingvalues(unit,oracle,mpmoracle)

def prototype(myfile,pack24,pack23,prune1,pack21,pack15,pack9,pack10,pack3,pack2,prune4):
    round1 = round1_computation(autolist,manuallist,pack24)
    round2 = round2_computation(autolist,manuallist,pack3)
    round3 = round3_computation(autolist,manuallist,pack9)
    round4 = round4_computation(autolist,manuallist,pack10)
    round5 = round5_computation(autolist,manuallist,pacak23)
    round6 = round6_computation(autolist,manuallist,pack21)
    round7 = round7_computation(autolist,manuallist,pack15)
    epic1 = round1 + round2 + round3 + round4 + round5 + round6 + round7
    epic1 = [x for x in epic1 if x not in prune1]
    epic1 = [x for x in epic1 if x not in prune4]
    epic1 = sortstuff(epic1)
    print(len(epic1))
    prototypecheck(oracle,mpmoracle,epic1)

def packs_n_prunes(myfile,testlen):
    pack2 = one2one_methods_plus_assertchecking_results(myfile,testlen)
    pack3 = one2one_asserts_results(myfile,testlen)
    pack9, pack10 = longestcommonsubsequence(myfile,testlen)
    pack15 = setmetrics_combo(myfile)
    pack21 = intersect_withassertsadditive_results(myfile)
    prune1 = scenarioskipgram(scenariocorpus,myfile,testlen)
    pack23 = tfidf_nfc(scenariocorpus,testlen,myfile)
    pack24 = tfidf_bnn(scenariocorpus,testlen,myfile)
    prune4 = LSI(myfile)
    return(myfile,pack24,pack23,prune1,pack21,pack15,pack9,pack10,pack3,pack2,prune4)

def XML_and_preprocessing(myfile):
    myfile,testlen,scenariocorpus,testcorpus,combinedcorpus = textpreprocessing(myfile)
    cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot = build_XML_trees()
    cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot = cleantrees(cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot)
    listofallfiles, myfile = processingXML(cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot,myfile)
    myfile = info_extraction_XML(myfile,listofallfiles)
    return myfile,cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot

def main():
    myfile = pd.read_csv(r"OptionBuilder.csv",header = 0)
    oracle, mpmoracle, oraclecluster, mpmoraclecluster = defineoracle_optionbuilder()
    myfile,cleanautotree,cleanautoroot,cleanmanualtree,cleanmanualroot = XML_and_preprocessing(myfile)
    #myfile,pack24,pack23,prune1,pack21,pack15,pack9,pack10,pack3,pack2,prune4 = packs_n_prunes(myfile,testlen)
    #manuallist,autolist = create_suitetype_lists(myfile)
    #prototype(myfile,pack24,pack23,prune1,pack21,pack15,pack9,pack10,pack3,pack2,prune4)

main()
# -*- coding: utf-8 -*-
import xlrd
import string
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
#import genism
from gensim import corpora, models
from gensim.models import CoherenceModel
import sys, errno
import xlsxwriter

#part of speech
tags=set(['CC','CD' ,'DT' ,'EX' ,'FW' ,'IN' ,'JJ' ,'JJR' ,'JJS' ,'MD' ,'PDT' ,'POS' ,'PRP' ,'PRP$' ,'RB' ,'RBR' ,'RBS' ,'UH' ,'VB' ,'VBD' ,'VBG' ,'VBN' ,'VBP' ,'VBZ' ,'WDT' ,'WP' ,'WP$' ,'WRB'])
tags2=set(['NN','NNS','NNP','NNPS'])

#dir corpus of brand-influencer set/ audiences set
paths=['D:\\aproject\\dataset_xls\\airline_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\auto_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\clothing_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\drink_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\electronics_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\entertainment_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\food_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\jewelry_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\makeup_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\nonprofit_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\services_brand_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\shoes_brand_post_v1.xlsx']

paths2=['D:\\aproject\\dataset_xls\\airline_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\auto_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\clothing_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\drink_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\electronics_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\entertainment_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\food_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\jewelry_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\makeup_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\nonprofit_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\services_in_post_v1.xlsx',
       'D:\\aproject\\dataset_xls\\shoes_in_post_v1.xlsx']
#load data
brand0=[]
brand=[]
influencer=[]
for path in paths:
    ExcelFile1=xlrd.open_workbook(path)
    sheet1=ExcelFile1.sheet_by_index(0)
    for i in range(0,sheet1.nrows):
        if(i!=0):
            if((i)%50==1 ):
                brand00=sheet1.cell(i,0).value.encode('utf-8').decode('utf-8-sig')
                brand0.append(brand00)

for path in paths2:
    ExcelFile1=xlrd.open_workbook(path)
    sheet1=ExcelFile1.sheet_by_index(0)
    for i in range(0,sheet1.nrows):
        
        if(i!=0):
            if((i)%50==1 ):
                brand1=sheet1.cell(i,0).value.encode('utf-8').decode('utf-8-sig')
                influencer1=sheet1.cell(i,1).value.encode('utf-8').decode('utf-8-sig')
                brand.append(brand1)
                influencer.append(influencer1)
#print(len(brand0))
#print(len(brand))

####################################################
# stop stop2 stop3 are stop words
stop = stopwords.words('english')
stop2=['pierre','anna','let','being','der','thomas','mama','www','link','day','today','year','look','de','part','max','code','email'
       ,'address','please','pakistani','franklin','http','click','person','contact','get','sale','way','share','canada','netherlands'
       ,'age','pant','paris','france','london','thing','detail','city','place','time','sale','germany','comment','pick','detail','cop','thanks'
       ,'tomorrow','aisa','tag','use','event','people','work','le','hour','country','tn','da','thank','one','godiva'
       ,'piece','night','morning','ford','pair','help','mini','product','brand','week','weekend','change','palace'
       ,'minite','toronto','project','head','create','tonight','asia','japan','talk','chance','afternoon','lot']    
stop3=[]
for line in open("D:\\aproject\\stopwords.txt"):
    line=line.lower()
    line = line.replace('\n', '')
    stop3.append(line)    
#####################################################
lemmatizer = WordNetLemmatizer()
account=''
docs=[]
docs_=[]
docs_2=[]
for path in paths:
    ExcelFile1=xlrd.open_workbook(path)
    sheet1=ExcelFile1.sheet_by_index(0)
    for i in range(0,sheet1.nrows):
        if(i!=0):
            try:
                text=sheet1.cell(i,3).value.encode('utf-8').decode('utf-8-sig')
            except:
                print(path,i)
            #docs.append(text)
            account=account+' '+text 
            if((i)%50==0 ):
                #print('00000000000000000000000')
                #print(account)
                docs.append(account)
                account=''

for path in paths2:
    ExcelFile1=xlrd.open_workbook(path)
    sheet1=ExcelFile1.sheet_by_index(0)
    for i in range(0,sheet1.nrows):
        if(i!=0):
            try:
                text=sheet1.cell(i,4).value.encode('utf-8').decode('utf-8-sig')
            except:
                print(path,i)
            #docs.append(text)
            account=account+' '+text 
            if((i)%50==0 ):
                docs.append(account)
                account=''
#print(len(docs))
#####################################################
dict={} # word frequencies
for dic in docs:
    doc = dic.lower()
    for c in string.punctuation: #去标点
        doc = doc.replace(c, ' ')
    for c in string.digits: #去数字
        doc = doc.replace(c, '')
    
    doc = nltk.word_tokenize(doc) #分割成单词
    doc_=[]
    for word in doc:#lower
        #print(word)
        word=word.lower()
        doc_ .append(word)
            
    # 只保留特定词性单词, 如名词
    pos_tags = nltk.pos_tag(doc_)
    doc_ = [w for w, pos in pos_tags if pos  in tags2]
    cleanDoc = []
    # 只保留长度不小于3的单词,去除停用词,验证是否为英文单词(利用wordnet)
    for word in doc_:
        word = lemmatizer.lemmatize(word) #词形还原
        cleanDoc.append(word)
    docs_.append(cleanDoc)
#################################################
index=0
for doc in docs_:
    for word in doc:
        #print(word)
        if word in dict.keys():
            dict[word]+=1
        else:
            dict[word]=1
#print(len(dict))

for key in dict.keys():
    if dict[key]<3:
        index+=1
#print(index)

for doc in docs_:
    doc_2=[]
    for word in doc:#去掉低频词
        #print(word)
        if ((dict[word]>2)and(word not in stop) and(word not in stop2) and(word not in stop3) and (len(word)>=3) and wordnet.synsets(word)):
            #word = lemmatizer.lemmatize(word)
            doc_2.append(word)
    docs_2.append(doc_2)
            
##############################################################
# 创建语料的词语词典，每个单独的词语都会被赋予一个索引
dictionary = corpora.Dictionary(docs_2)
dictionary.save("D:\\aproject\\dictionary1.dic")
#print(dictionary)
# 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
corpus = [dictionary.doc2bow(doc) for doc in docs_2]
#print(corpus)
corpora.MmCorpus.serialize("D:\\aproject\\corpus1.m", corpus)
##############################################################








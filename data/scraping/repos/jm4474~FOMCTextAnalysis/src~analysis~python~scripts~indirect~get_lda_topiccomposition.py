#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: olivergiesecke
1) Collect the data on the speakers and text for each alternative.
2) Do the regular pre-processing for each text entry.
3) Apply standard LDA
4) Provide summary statics how the probability mass lines up with the different alternatives.
5) Check alignment with the voting record.
"""
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
import itertools  
import os
import gensim
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
import matplotlib
import matplotlib.pyplot as plt
import re
import seaborn as sns


import create_lda_data

import provide_helperfunctions

from nltk.util import ngrams
from collections import Counter   
from pprint import pprint 
from gensim.models.coherencemodel import CoherenceModel
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
    
# Set random state for entire file
rnd_state=5

###############################################################################

    ### Import data ###
data = create_lda_data.main()
data.rename(columns={"start_date":"date"},inplace=True)
data.to_csv("../output/lda_dataset.csv",index=False)
    ### Data Selection  ###
data['new']=1
df_balt=data[data['d_alt']==1].pivot_table(index="date",values='new',aggfunc=np.sum).reset_index()
df_summary = data.pivot_table(index="date",values='new',columns=['d_alt','votingmember'],aggfunc=np.sum)

# Keep only dates for which alternatives are available and speakers who are votingmembers
data_speakers=data[data['votingmember']==1].merge(df_balt,on='date',how='inner')
data_alternatives=data[data['d_alt']==1]
data_alternatives = data_alternatives[data_alternatives['content']!='[]'].copy()
#data_alternatives.to_csv("~/Desktop/alternativetext.csv")


    ### Check the coverage of the speaker data ###
alt_dates = pd.DataFrame(data[data['d_alt']==1].date.unique()).rename(columns={0:"date"})
alt_dates['alt']=1
date_speakers = pd.DataFrame(data[data['votingmember']==1].date.unique()).rename(columns={0:"date"})
date_speakers['speaker']=1
merge_df = pd.merge(alt_dates,date_speakers,on="date",how="outer")


print("Number of alternative dates: %d" % len(data_alternatives['date'].unique()))
print(f"Earliest meeting with alternatives: {data_alternatives['date'].min()}" )
print(f"Latest meeting with alternatives: {data_alternatives['date'].max()}" )


print("Number of speaker dates: %d" % len(data_speakers['date'].unique()))
print("Earliest date of speaker: %s" % data_speakers['date'].min())
print("Latest date of speaker: %s" % data_speakers['date'].max())

print("Number of words for the speakers is: {:.3f} million".format(len(" ".join(data_speakers['content'].tolist())) / 1e6))
print("Number of words for the alternatives is: {:.3f} million".format(len(" ".join(data_alternatives['content'].tolist())) / 1e6 ))

    ### Summary Statistics ###
with open("../output/file_basic_sumstats.tex","w") as file:
    file.write("DOCUMENTS COLLECTED:\\\\\\\\")
    file.write(f"Number of alternative dates: \t \t {len(data_alternatives['date'].unique())}\\\\")
    file.write(f"Earliest meeting with alternatives:\t \t {data_alternatives['date'].min()} \\\\")
    file.write(f"Latest meeting with alternatives:\t \t {data_alternatives['date'].max()} \\\\ \\\\" )
    
    file.write(f"Number of speaker dates: {len(data_speakers['date'].unique())}\\\\")
    file.write(f"Earliest date of speaker: {data_speakers['date'].min()}\\\\")
    file.write(f"Latest date of speaker: {data_speakers['date'].max()}\\\\\\\\")
    
    file.write("Number of words for the speakers is: {:.3f} million \\\\".format(len(" ".join(data_speakers['content'].tolist())) / 1e6))
    file.write("Number of words for the alternatives is: {:.3f} million \\".format(len(" ".join(data_alternatives['content'].tolist())) / 1e6 ))
    


# =============================================================================
#     # Subsample the speakers -- only to learn the model
# data_speakers_subsample = data_speakers.sample(frac =.1 ,random_state=5) 
# print("Number of words for the subsample of speakers is: %s" % (len(" ".join(data_speakers_subsample ['content'].tolist())) / 1e6))
# data_sel = pd.concat([data_speakers_subsample,data_alternatives],axis=0, join='inner')
# data_sel = data_sel.reset_index()
# =============================================================================

    ### Learn the model based only on basis of the alternatives ###
print("\n### MODEL ESTIMATION - ALTERNATIVES ONLY ###\n")
data_sel = data_alternatives.reset_index()

    # Do simple preprocessing
data_sel['parsed']=data_sel['content'].apply(provide_helperfunctions.extract_token)
data_sel['parsed'].loc[1]

    ### Revome stopwords and do stemming ###
stopwordsnltk = stopwords.words('english')
stopwordsnltk.extend(["mr","chairman","yes",'restrict', 'control','class','page',
                      'chart','strictli',"presid", "governor", "would","think",
                      "altern","could","committe","may",
                      "ty","yt","πt","bt","yt","na","na","gt","row","qiv","rtc","tip","dec","jul",
                      "confid","interv","ut","seven","confidenti","jun",
                      "jan","feb","mar","apr","aug","sep","oct","nov",'march','septemb','fr','june','april','nan'])

data_sel['parsed_cleaned']=data_sel['parsed'].apply(lambda x: 
    provide_helperfunctions.remove_stopwords(
            provide_helperfunctions.do_stemming(
                    provide_helperfunctions.remove_stopwords(x,stopwordsnltk)),stopwordsnltk))

    ### Build corpus ###
texts=[]
for row_index,row in data_sel.iterrows():
    item=row['parsed_cleaned']
    texts.append(item)    

    ### Extract tokens ###
tokens =[]
for text in texts:
    for word in text:
        tokens.append(word)

    ### Extract the top 100 common tokens ###
counter = Counter(tokens)
n_topwords=100
provide_helperfunctions.plot_wordlist(counter.most_common(n_topwords),n_topwords,n_percolumns=34,filename="../output/tab_tf_alternatives.tex")

    ### Extract the top 100 bigrams tokens ###
bi_grams = list(ngrams(tokens, 2)) 
counter = Counter(bi_grams)
n_topwords=100
provide_helperfunctions.plot_wordlist(counter.most_common(n_topwords),n_topwords,n_percolumns=34,filename="../output/tab_tf_bigrams.tex")
    ### Add bi-grams ###
n_bigrams = 100
bi_gram_mostcommon = ["_".join(ele[0]) for ele in counter.most_common(n_bigrams)]
texts = provide_helperfunctions.add_bigrams(texts,bi_gram_mostcommon)

    ### Extract the top 100 trigrams tokens ###
tri_grams = list(ngrams(tokens, 3)) 
counter = Counter(tri_grams)
n_topwords = 68
provide_helperfunctions.plot_wordlist(counter.most_common(n_topwords),n_topwords,n_percolumns=34,filename="../output/tab_tf_trigrams.tex")
    ### Add tri-grams ###
n_tri_grams = 50
tri_gram_mostcommon = ["_".join(ele[0]) for ele in counter.most_common(n_tri_grams)]
texts = provide_helperfunctions.add_trigrams(texts,tri_gram_mostcommon)

    ### Plot TF-IDF figure to decide on the terms ###
tokens =[]
for text in texts:
    for word in text:
        tokens.append(word)
# Unique words
unique_tokens =sorted(list(set(tokens)))

tf_idf = provide_helperfunctions.get_tdidf(tokens,unique_tokens,texts)
tf_idf_sort =np.sort(tf_idf)
tf_idf_invsort = tf_idf_sort[::-1]  

plt.figure(figsize=(12,7))
plt.plot(np.arange(len(unique_tokens)),tf_idf_invsort)
plt.ylabel('Tf-idf weight')
plt.xlabel('Rank of terms ordered by tf-idf')
plt.savefig('../output/fig_alt_tfidf.pdf')

# print terms with the largest ranking
def merge(list1, list2):       
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 

n_topwords = 68
indices = tf_idf.argsort()[-n_topwords:][::-1]
tfidf_top = tf_idf[indices]
word_arr = np.asarray(unique_tokens)
word_top= word_arr[indices]
counter = merge(list(word_top),list(tfidf_top))
provide_helperfunctions.plot_wordlist(counter,n_topwords,n_percolumns=34,filename="../output/tab_tfidf_list.tex",columnnames=['#','term','tf-idf score'])
   
    ### Keep top x words ###
totaln_words=2200
texts =  provide_helperfunctions.trim_texts(tf_idf,unique_tokens,texts,totaln_words)

    ### Build dictionary ###
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

###############################################################################
    ### Model Selection ###
run_modelsel = False   
run_numtopic = False

if run_modelsel == True:    
        ### Explore multiple dimension of the parameter space - TAKES A LONG TIME ###
    alpha_v = np.array([0.001,0.01,0.1,0.5,1])
    eta_v = np.array([0.001,0.01,0.1,0.5,1])
    topic_v =np.array([5,10,15,20])
    models_df  = provide_helperfunctions.explore_parameterspace(totaln_words,corpus,dictionary,rnd_state,texts,alpha_v,eta_v,topic_v)
    models_df=models_df.sort_values(by='coherence score (PMI)',ascending=False).reset_index().drop(columns="index")
    models_df['model']=models_df.index
    models_df['model']=models_df['model'].apply(lambda x:x+1)
    models_df.to_latex("../output/tab_models.tex",index=False,float_format="%.3f")
    
    # plot the parameter space
    #provide_helperfunctions.plot_parameterspace(models_df)
    
    # extract the parameter values for the highest coherence score
    row  = models_df.loc[models_df['coherence score (PMI)'].idxmax()]
    row.to_pickle("../output/opt_parameter_coh")
    row  = models_df.loc[models_df['perplexity'].idxmax()]
    row.to_pickle("../output/opt_parameter_perplexity")

row = pd.read_pickle("../output/opt_parameter_coh")
num_topics = int(row['# topics'])
eta_p = row['eta']
alpha_p = row['alpha']

provide_helperfunctions.output_number(num_topics,filename="../output/par_bs_numtopoics.tex",dec=0)
provide_helperfunctions.output_number(eta_p,filename="../output/par_bs_eta.tex",dec=3)
provide_helperfunctions.output_number(alpha_p,filename="../output/par_bs_alpha.tex",dec=3)

if run_numtopic == True:
    ### Number of topic evaluations ###
    eta = eta_p
    alpha = alpha_p
    provide_helperfunctions.explore_numberoftopics(totaln_words,corpus,dictionary,texts,rnd_state, eta , alpha )
    provide_helperfunctions.output_number(eta,filename="../output/par_topic_eta.tex",dec=3)
    provide_helperfunctions.output_number(alpha,filename="../output/par_topic_alpha.tex",dec=3)

###############################################################################
    ### Model Estimation ###

    ### Do LDA ###
print(f"# Model parameter: Number of topics = {num_topics}, eta = {eta_p}, alpha = {alpha_p} random state = {rnd_state}\n")
ldamodel = models.ldamodel.LdaModel(corpus, num_topics, id2word = dictionary, passes=30 ,eta=eta_p ,alpha = alpha_p, random_state=rnd_state)
# Perplexity
logperplexity = ldamodel.log_perplexity(corpus, total_docs=None)
provide_helperfunctions.output_number(logperplexity,filename="../output/par_logperplexity.tex",dec=3)

# Coherence measure 
cm = CoherenceModel(model=ldamodel, corpus=corpus, texts = texts, coherence='c_uci') # this is the pointwise mutual info measure.
coherence = cm.get_coherence()  # get coherence value
provide_helperfunctions.output_number(coherence,filename="../output/par_coherence.tex",dec=3)

    ### Inspect the topics ###
n_words=10
x=ldamodel.show_topics(num_topics, num_words=n_words,formatted=False)
topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

print("# These are the topic distributions for the estimated model:\n")
for topic,words in topics_words:
    print(str(topic)+ "::"+ str(words))
    
    ### Visualize as a heatmap ###
provide_helperfunctions.draw_heatmap(x,n_words,params=(num_topics,eta_p,alpha_p), pmin = 0, pmax = 0.05)



###############################################################################

data = pd.concat([data_speakers,data_alternatives],axis=0, join='inner')
data = data.reset_index()
data_light = data[['d_alt','date', 'speaker', 'speaker_id', 'votingmember', 'ambdiss','tighterdiss', 'easierdiss']]

    # Do simple preprocessing
data['parsed']=data['content'].apply(provide_helperfunctions.extract_token)

    # Revome stopwords and do stemming
data['parsed_cleaned']=data['parsed'].apply(lambda x: 
    provide_helperfunctions.remove_stopwords(
            provide_helperfunctions.do_stemming(
                    provide_helperfunctions.remove_stopwords(x,stopwordsnltk)),stopwordsnltk))

    ### Build corpus ###
texts=[]
for row_index,row in data.iterrows():
    item=row['parsed_cleaned']
    texts.append(item)    

    ### Add bigrams and trigrams ###
texts = provide_helperfunctions.add_bigrams(texts,bi_gram_mostcommon)
texts = provide_helperfunctions.add_trigrams(texts,tri_gram_mostcommon)

    ### Build the dictionary ###
corpus = [dictionary.doc2bow(text) for text in texts]

    ### Extract topic vectors ###
sent_topics_df = provide_helperfunctions.extract_vectors(ldamodel,int(num_topics),corpus)
data_lda =  pd.concat([data,sent_topics_df],axis=1, join='inner')

    ### Apply SVD for dimensionality reduction ##
print("\n### DIMENSIONALITY REDUCTION FOR VISUAL OUTPUT ###\n")
col_topics = [ col for col in data_lda.columns if re.match("^topic",col)]
dfvalues=data_lda[col_topics].values
twodim = provide_helperfunctions.reduce_to_k_dim(dfvalues)
df_pca=pd.DataFrame(twodim)
df_pca.rename(columns={0:'PCI1',1:'PCI2'},inplace=True)
data_lda_pca = pd.concat([data_lda,df_pca],axis=1, join='inner')
data_lda_pca.sort_values(by="date",inplace=True)

    ### Compute the preferred alternative for each speaker and contrast it with the voting outcome

    # Individual date for all speakers
date = '1990-10-02'
print(F"\n### CONFUSION MATRIX FOR {date} - HELLINGER DISTANCE ###\n")
pref_distance = provide_helperfunctions.create_distance( date , data_lda_pca , col_topics )
col_alts = [ col for col in pref_distance.columns if re.match("^alt",col)]
pref_distance["pred_vote"] = pref_distance[col_alts].idxmin(axis=1)
pref_distance = pref_distance.merge(data_lda_pca[['date','speaker','act_vote']],on=['speaker','date'])
confusion_matrix = pd.crosstab(pref_distance["act_vote"], pref_distance["pred_vote"], rownames=['Actual'], colnames=['Predicted'])
dataexample = data_lda_pca[(data_lda_pca['d_alt']==1) | (data_lda_pca['votingmember']==1)][data_lda_pca['date']==date]
pref_distance.drop(columns="date",inplace=True)
print(pref_distance)
provide_helperfunctions.output_plot(date,dataexample)
    
    # Latex output
data_lda_pca.rename(columns=dict(zip(col_topics,[f"t_{i+1}" for i in range(num_topics)]))).loc[data_lda_pca['date']==date,['speaker']+[f"t_{i+1}" for i in range(num_topics)]].sort_values(by="speaker").to_latex(f"../output/tab_topicdist_{date}.tex",index=False,float_format="%.2f")
pref_distance.rename(columns=dict(zip(col_alts,[f"hd_{col}" for col in col_alts]))).to_latex(f"../output/tab_pref_matrix_{date}.tex",index=False,float_format="%.2f")
confusion_matrix.reset_index().rename(columns={"Actual":"Actual \  Predicted"}).to_latex(f"../output/tab_conf_matrix_{date}.tex",index=False,float_format="%.2f")

    # All dates and all speakers 
print("\n### CONFUSION MATRIX ALL DATA - HELLINGER DISTANCE ###\n")
pref_distance = pd.DataFrame()
for date in data_lda_pca['date'].unique():
    help_df = provide_helperfunctions.create_distance(date,data_lda_pca, col_topics )
    pref_distance = pd.concat([pref_distance,help_df])    

pref_distance["pred_vote"] = pref_distance[col_alts].idxmin(axis=1)
pref_distance = pref_distance.merge(data_lda_pca[['date','speaker','act_vote']],on=['speaker','date'])
confusion_matrix = pd.crosstab(pref_distance["act_vote"], pref_distance["pred_vote"], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)


    ### Show a few examples ###
    
print("\n### VISUAL OUTPUT ###\n")

for date in ['2005-08-09','1993-05-18','1997-05-20','2000-11-15']:
    #date='1991-10-01'
    dataexample = data_lda_pca[(data_lda_pca['d_alt']==1) | (data_lda_pca['votingmember']==1)][data_lda_pca['date']==date]
    #print(dataexample[["speaker"]+col_topics+['PCI1','PCI2']])
    provide_helperfunctions.output_plot(date,dataexample)
    
    
    # Greenspan
print("\n### GREENSPAN - HELLINGER DISTANCE ###\n")
pref_distance = pd.DataFrame()
for date in data_lda_pca['date'].unique():
    help_df = provide_helperfunctions.create_distance(date,data_lda_pca[(data_lda_pca['speaker']=="greenspan") | (data_lda_pca['d_alt']==1)],col_topics  )
    pref_distance = pd.concat([pref_distance,help_df])   
    
pref_distance["newdate"]=pd.to_datetime(pref_distance['date'])
pref_distance = pref_distance.merge(data_lda_pca[['date','speaker','act_vote']],on=['speaker','date'],how="left")    

pref_distance=pref_distance.reset_index()
pref_distance['ch_alt']=np.nan
for idx,row in pref_distance.iterrows():
    vote = row['act_vote']
    print(vote)
    pref_distance.loc[idx,'ch_alt'] =  row[vote]
    
fig = plt.figure(figsize=(15, 8))
#fig.suptitle("Greenspan Hellinger Distance")
ax = fig.add_subplot(1, 1, 1)
pref_distance.sort_values(by="date",inplace=True)
ax.plot(pref_distance['newdate'], pref_distance['alta'], 'ro',markersize=2)
ax.plot(pref_distance['newdate'], pref_distance['altb'], 'go',markersize=2)
ax.plot(pref_distance['newdate'], pref_distance['altc'], 'yo',markersize=2)
ax.plot(pref_distance['newdate'], pref_distance['altd'], 'mo',markersize=2)
ax.plot(pref_distance['newdate'], pref_distance['ch_alt'], 'o', markersize=6,fillstyle='none',label="alt chosen")
ax.legend()
plt.savefig("../output/fig_greenspan_hd.pdf")

    ### Push results to Overleaf
# os.system("cp par* fig* tab* *.tex ~/Dropbox/Apps/Overleaf/FOMC_Summer2019/files/")

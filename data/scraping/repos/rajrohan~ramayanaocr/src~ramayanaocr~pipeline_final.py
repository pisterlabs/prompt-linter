import warnings
warnings.filterwarnings("ignore")
import string
import numpy as np
import pandas as pd
from collections import Counter
import ast
import re

from tqdm import tqdm

import nltk
from nltk.tag import tnt
from nltk.corpus import stopwords
from nltk.corpus import indian
from nltk.tokenize import word_tokenize,sent_tokenize

import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.summarization import summarize
import gensim.corpora as corpora
from gensim.test.utils import datapath

from flair.data import Sentence
from flair.models import SequenceTagger

import stanfordnlp

from helper import create_str_from_list,create_match_list_n_log,clean_text,store_file
from helper import create_list_remove_number,textrankTfIdf,orderSentences,extract_ner




# from bokeh.plotting import figure, output_file, show
# from bokeh.models import Label
# from bokeh.io import output_notebook
# output_notebook()

#get_ipython().run_line_magic('matplotlib', 'inline')

text_sentence = []

nlp = stanfordnlp.Pipeline ( lang = 'hi' ) 

def load_input_txt(datafile):
    # Read all seprate text file to one list
    with open(datafile,'r',encoding='utf-8') as f:
        text = f.read()
        text = text.split("॥")
    return text

text = load_input_txt('./data/hindi.txt')
text_sentence = create_list_remove_number(text)
str_text = create_str_from_list(text_sentence)

def gensim_summarization(str_text):
    startflag = 0
    endflag = 2500
    parsed_text = {'text':[], 'title':[]}
    for summary in range(int(len(str_text)/endflag)):
        summarization_block = summarize(str_text[startflag:endflag])
        parsed_text['text'].append(str_text[startflag:endflag])
        parsed_text['title'].append(summarization_block)
        startflag +=2500
        endflag +=2500
    return pd.DataFrame(parsed_text)
# Gensim Summarization
gensim_summarize_df = gensim_summarization(text)

### Alternative Summarization

def alternate_summarization(text_sentence):
    startflag = 0
    endflag = 25
    parsed_text = {'text':[], 'alt_title':[]}
    sentence =''
    for summary in range(int(len(text_sentence)/endflag)):
        rankedSentences = textrankTfIdf(text_sentence[startflag:endflag])
        orderedsentences = orderSentences(rankedSentences, text_sentence[startflag:endflag], text_sentence[startflag:endflag])
        for ordered in orderedsentences:
            if ordered != "":
                sentence += ordered 
        parsed_text['alt_title'].append(sentence)
        parsed_text['text'].append(text_sentence[startflag:endflag])
        startflag +=25
        endflag +=25
        sentence =''
    return pd.DataFrame(parsed_text)


alternate_summarization = alternate_summarization(text_sentence)
#df2.to_csv('summary_alt_gensim.csv')


#df2 = pd.read_csv('../data/summary_alt_gensim.csv')


# load the model you trained
def ner_tag(summarize_df):
    model = SequenceTagger.load('./data/netag/best-model.pt')
    df_ner = pd.DataFrame()
    r, c = summarize_df.shape
    for row_no in range(r):
        sentence = Sentence(summarize_df['alt_title'][row_no])
        model.predict(sentence)
        sentence_ner = sentence.to_tagged_string()
        temp_df = extract_ner(sentence_ner,row_no)
        df_ner = df_ner.append(temp_df,ignore_index=True)
    return df_ner

df_ner = ner_tag(alternate_summarization)

#df_ner.to_csv('../data/ner_tag.csv')
#df_ner = pd.read_csv('../data/ner_tag.csv',index_col=0)

# Get names of indexes for which column Age has value 30
df_ner_imp = df_ner[ df_ner['ner'].str.endswith('PERSON') | df_ner['ner'].str.endswith('LOCATION') ]
# Delete these row indexes from dataFrame
#df_ner.drop(indexNames , inplace=True)
#df_ner_imp.to_csv('../data/ner_tag_imp.csv')
#df_ner.to_csv('../data/ner_tag.csv')

# Standford nlp is being used for lemmatization


# In[95]:
# hindi stopwords
stop_words_df = pd.read_csv("./data/stopwords.txt", header = None)
stop_words = list(set(stop_words_df.values.reshape(1,-1).tolist()[0]))
stop_words.extend(["।", "।।", ")", "(", ",",'"',"हे", "हो", 'में','से','COMMA'])

def lemmatization(text):
    lemmatized_text = []
    for line in tqdm(text):
        if line not in [""," "] :
            doc = nlp(line)
            for sent in doc.sentences:
                for wrd in sent.words:
                    #extract text and lemma
                    lemmatized_text.append(wrd.lemma)
    return lemmatized_text

def remove_stopwords(word_tokenized,stop_words):
    return [word for word in word_tokenized if word not in stop_words]

def custom_remove_garbage(original_words_list,list_of_garbage_words):
    tmp_list = [word for word in original_words_list if word not in list_of_garbage_words] # garbage list
    tmp_list = [word for word in tmp_list if len(re.findall("\d+",word))==0] # english numbers
    tmp_list = [word for word in tmp_list if len(re.findall("[a-zA-Z]+",word))==0] # english alphabets
    return tmp_list

def Diff(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif 


# ## Topic Meddling after prepocessing

# In[149]:


def topic_modelling(text_sentence,stop_words):
    df = pd.DataFrame()
    startflag = 0
    endflag = 25
    for text in range(int(len(text_sentence)/endflag)):
        
        #Clean text after lemmatization
        lemmatized = lemmatization(text_sentence[startflag:endflag])
        clean_text = remove_stopwords(lemmatized,stop_words)
        #len(clean_text),len(lemmatized) #281,612
        
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in clean_text]
        #print(stripped[:25])
        #print(len(clean_text),len(stripped)) #281

        resultant = Diff(clean_text,stripped)
        #len(set(resultant)) #19

        final_text = custom_remove_garbage(stripped,resultant)
        #len(final_text) #265
        
        # Create Dictionary
        id2word = corpora.Dictionary([final_text])

        # Create Corpus
        texts = final_text

        # Term Document Frequency
        corpus = [id2word.doc2bow(texts)] 
        
        # Build LDA model

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=1, 
                                           random_state=100,
                                           update_every=10,
                                           chunksize=100,
                                           passes=20,
                                           alpha='auto',
                                           per_word_topics=True)
        #pprint(lda_model.print_topics())
        temp_file = datapath("model")
        lda_model.save(temp_file)
        lda = gensim.models.ldamodel.LdaModel.load(temp_file)
        lda.update(corpus)
        lda.save('model')
        temp_df = pd.DataFrame(lda.print_topics())
        df = df.append(temp_df,ignore_index=True) 
        startflag +=25
        endflag +=25
    return df

combined_model_topic_df = topic_modelling(text_sentence,stop_words)

#df4.to_csv('combined_model_topic.csv')


# In[3]:


#df4 =pd.read_csv('../data/combined_model_topic.csv',index_col=0)

#re.findall('[!@#$%^&*(),.?":{}|<>]',df4['1'][0])

#re.sub('[!@#$%^&*(),.?"+:{}|<>0-9]', '', df4['1'][0])


def process_topic(df):
    parsed_text = {'top_topic':[]}
    
    for ind in df.index:
        parsed_text['top_topic'].append(re.sub('[!@#$%^&*(),.?"+:{}|<>0-9]', '', df[1][ind]))
    return pd.DataFrame(parsed_text)    


topic_df= process_topic(combined_model_topic_df)

topic_df.to_csv('./data/output/combined_model_topic_top_clean.csv')



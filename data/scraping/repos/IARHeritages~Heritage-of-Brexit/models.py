'''
### The code was developed as part of the Ancient Identities in Modern Britain (IARH) project
### It allows performing topic modelling on a corpus of text
### Project: Ancient Identities in Modern Britain (IARH); ancientidentities.org
### Author: Mark Altaweel
'''

import os
import warnings
import gensim

import sys
import csv
from nltk.tokenize import RegexpTokenizer
import pyLDAvis.gensim

from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
from gensim.utils import lemmatize
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
stops = set(stopwords.words('english'))  # nltk stopwords list


'''
    Method to get the text output from results in a CSV. Retrieves relevant texts only.
    @param pn the path to find the relevant text
    '''
def retrieveText(self,pn):
    del self.listResults[:]
        
    doc_set=[]
    os.chdir(pn+'/output')
    en_stop = stops
    result=[]
    for filename in os.listdir(os.getcwd()):
        txt=''
        if(filename == ".DS_Store" or "lda" in filename or "hdp" in filename or ".csv" not in filename):
            continue
        print(filename)
        with open(filename, 'rU') as csvfile:
            reader = csv.reader(csvfile, quotechar='|') 
                
            i=0
            try:
                for row in reader:
                    if row in ['\n', '\r\n']:
                        continue;
                    if(i==0):
                        i=i+1
                        continue
                    if(len(row)<1):
                        continue

                    text=''
                    for r in row:
                        text+=r.strip()
                            
                    text=re.sub('"','',text)
                    text=re.sub(',','',text)
                    text.strip()
                    tFalse=True
                        
                    if(len(result)==0):
                        result.append(text)
                        i+=1
                        txt=txt+" "+text
                            
                    if(tFalse==True):
                            txt=txt+" "+text
                             
                            if text==' ':
                                continue
                             
                            tokenizer = RegexpTokenizer(r'\w+')
                            text = tokenizer.tokenize(unicode(text, errors='replace'))
                            stopped_tokens = [t for t in text if not t in en_stop]
                             
                            doc_set.append(stopped_tokens)  
                    i+=1 
            except csv.Error, e:
                sys.exit('line %d: %s' % (reader.line_num, e))
            
            
            
        return doc_set

'''
Process text files based on minimum length of file

@param files for processing
'''
def preProcsText(files):
  
        for f in files:
            yield gensim.utils.simple_preprocess(f, deacc=True, min_len=3)

'''
Method for text processing with input text.

@param input texts
@return texts the texts to be returned after processing.
'''
def process_texts(texts):
    """
    Function to process texts. Following are the steps we take:
    
    1. Stopword Removal.
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).
    
    Parameters:
    ----------
    texts: Tokenized texts.
    
    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """
    
    # reg. expression tokenizer
        
    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [bigram[line] for line in texts]
    texts = [[word.split('/')[0] for word in lemmatize(' '.join(line), allowed_tags=re.compile('(NN)'), min_length=3)] for line in texts]

    return texts

'''
Method for using a coherence model to look at topic coherence for LDA models.

@param dictionary the dictionary of assessment 
@param corpus the texts
@param limit the limit of topics to assess
@return lm_list lda model output
@return c_v coherence score
''' 
def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        del cm
            
    return lm_list, c_v


'''
The below code is executed to conduct the models for topic modelling and 
coherence testing for LDA models
'''
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]
results=retrieveText(pn)

bigram = gensim.models.Phrases(results) 
#train_texts = process_texts(train_texts)

train_texts=process_texts(results)

dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]

#up to 50 topics are tested 
for i in range(1,50,1):
   
    #lda model
    ldamodel = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary)
    num=str(i)
    ldamodel.save('lda'+num+'.model')
    ldatopics = ldamodel.show_topics(num_topics=i)
    
    visualisation = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization_'+str(i)+'_.html') 

i=50

#coherence model evaluation
lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=train_texts, limit=i)


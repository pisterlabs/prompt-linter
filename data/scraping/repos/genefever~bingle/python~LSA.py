import os
import matplotlib.pyplot as plt
import re
import pandas as pd
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag,FreqDist
from spellchecker import SpellChecker

def remove_whitespace(text):
    return  " ".join(text.split())

def spell_check(text):    
    result = []
    spell = SpellChecker()
    for word in text:
        correct_word = spell.correction(word)
        result.append(correct_word)    
    return result   

def remove_stopwords(text):
    result = []
    for token in text:
        if token not in en_stopwords:
            result.append(token)            
    return result

def remove_punct(text):
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst

def frequent_words(df):
    lst=[]
    for text in df.values:
        lst+=text[0]
    fdist=FreqDist(lst)
    return fdist.most_common(10)

def lemmatization(text):    
    result=[]
    wordnet = WordNetLemmatizer()
    for token,tag in pos_tag(text):
        pos=tag[0].lower()        
        if pos not in ['a', 'r', 'n', 'v']:
            pos='n'            
        result.append(wordnet.lemmatize(token,pos))
    return result

def stemming(text):
    porter = PorterStemmer()    
    result=[]
    for word in text:
        result.append(porter.stem(word))
    return result

def prepare_corpus(doc_clean):
    """
    Input  : clean document
    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    Output : term dictionary and Document Term Matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    return dictionary,doc_term_matrix

def create_gensim_lsa_model(doc_clean,number_of_topics,words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    lsastr = lsamodel.print_topics(num_topics=number_of_topics, num_words=words)
    lst = (str(i[1]).rstrip('\x00') for i in lsastr)
    idx = 0
    for j in lst: 
        print("Topic" + str(idx) + " :  " + j)    
        idx +=1
    return lsamodel

def compute_coherence_values(dictionary, number_of_topics, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def plot_graph(doc_clean, number_of_topics, start, stop, step):
    dictionary, doc_term_matrix=prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, number_of_topics, doc_term_matrix,doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    
def main():
    # LSA Model
    number_of_topics=10
    words=10
    os.getcwd()
    os.listdir()
    header_list = ["ID", "Content"]
    df=pd.read_csv('top500.csv', encoding='utf-8', names=header_list)
    
    df_text=df[['Content']]
    #print(df_text.head())

    df_text['Content']=df_text['Content'].str.lower()
    #print(df_text.head())

    df_text['Content']=df_text['Content'].apply(lambda X: word_tokenize(X))
    #print(df_text.head())

    df_text['Content'] = df_text['Content'].apply(spell_check)
    #print(df_text.head())

    df_text['Content'] = df_text['Content'].apply(remove_stopwords)
    #print(df_text.head())

    text=df_text['Content'][0] 
    df_text['Content'] = df_text['Content'].apply(remove_punct)

    freq_words = frequent_words(df_text)
    lst = []
    for a,b in freq_words:
        lst.append(b)
    def remove_freq_words(text):
        result=[]
        for item in text:
            if item not in lst:
                result.append(item)
        return result   
    df_text['Content']=df_text['Content'].apply(remove_freq_words)
    df_text['Content']=df_text['Content'].apply(lemmatization)
    df_text['Content']=df_text['Content'].apply(stemming)    
    
    model=create_gensim_lsa_model(df_text['Content'],number_of_topics,words)

    #start,stop,step=2,12,1
    #plot_graph(df_text['Content'], number_of_topics, start,stop,step)

if __name__ == '__main__':
    main()
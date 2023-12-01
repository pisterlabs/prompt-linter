#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:49:55 2019

@author: github.com/sahandv
"""
import io
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def union_lists(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list 

def tokenize_series(data):
    import pandas as pd
    from tqdm import tqdm
    from nltk.tokenize import word_tokenize

    tqdm.pandas()
    
    result = data.progress_apply(lambda x: word_tokenize(x))
    result = pd.DataFrame(result)
    
    return result

def tokenize_series_fast(data,delimiter=None,flatten=False): # Not accurate, but fast :)
    import pandas as pd
    from tqdm import tqdm

    tqdm.pandas()
    
    if delimiter is None:
        result = data.progress_apply(lambda x: str(x).split())
    else:
        result = data.progress_apply(lambda x: str(x).split(delimiter))
        
    result = pd.DataFrame(result)

    if flatten is True:
        flatten = []
        for index, row in tqdm(result.iterrows(),total = result.shape[0]):
            flatten.extend(row[data.name]) #= flatten+list(row['abstract'])
            
            
    flatten = pd.DataFrame(flatten,columns=['tokens'])

    return result,flatten


def corpus_tokenize(sentences):
# =============================================================================
#     Tokenize corpus by gensim
# =============================================================================
    import gensim
    for sentence in tqdm(sentences,len(sentences)):
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def lemmatization_spacy(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
# =============================================================================
#     """https://spacy.io/api/annotation"""
# =============================================================================
    import spacy
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def indexed_text_to_string(indexed_text_dict,text_length):
    """
    Parameters
    ----------
    indexed_text_dict : Dict
        Indexed text dictionary
    text_length : Int

    Returns
    -------
    String
        Concatenated string
    """
    import numpy as np

    string_text = np.array(['']*(text_length),dtype='U100')
    for key in indexed_text_dict.keys():
        string_text[indexed_text_dict[key]] = key
    return ' '.join(list(string_text))


def get_wordnet_pos(word,download_nltk=False):
    """Map POS tag to first character lemmatize() accepts"""
    import nltk
    if download_nltk is True:
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
    from nltk.corpus import wordnet
    
    
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def string_pre_processing(input_str,stemming_method='PorterStemmer',stop_words_extra=None,
                          punctuation_removal=True,lowercase=True,tag_removal=True,special_char_removal=True,
                          stop_word_removal=True,lemmatization=False,download_nltk=True,verbose=False,pos=False):
# =============================================================================
# NLTK text pre processing helper
# =============================================================================
    if download_nltk is True:
        import nltk
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
    import re
    from nltk.stem import PorterStemmer, LancasterStemmer
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

# Lower case 
    if lowercase is True:
        input_str = input_str.lower()
        if verbose is True: print('\nCase lowered')
    
# Remove punctuations
    if punctuation_removal is True:
        input_str = re.sub('[^a-zA-Z]', ' ', input_str)
        if verbose is True: print('\nRemoved punctuations')
    
# Remove tags
    if tag_removal is True:
        input_str=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",input_str)
        if verbose is True: print('\nRemoved tags')
    
# Remove special characters and digits
    if special_char_removal is True:
        input_str=re.sub("(\\d|\\W)+"," ",input_str)
        if verbose is True: print('\nRemoved special characters and digits')
        
# Tokenize
    input_str=word_tokenize(input_str)

# Stop word removal
    if stop_word_removal is True:
        stop_words = set(stopwords.words("english"))
        new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "shown"]
        stop_words = stop_words.union(new_words)
        if stop_words_extra!=None:
            stop_words = stop_words.union(stop_words_extra)
        input_str = [word for word in input_str if not word in stop_words] 
            
        
# Lemmatization stage
    if lemmatization is not False:
        if verbose is True: print("\nPerforming lemmatization on ",input_str)
        lemmatizer=WordNetLemmatizer()
#        input_str = [lemmatizer.lemmatize(word) for word in input_str]
        if lemmatization=='ALL':
            input_str = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in input_str]
        if lemmatization=='DEF':
            input_str = [lemmatizer.lemmatize(w) for w in input_str]
        else:
            input_str = [lemmatizer.lemmatize(w, lemmatization) for w in input_str]
# Stemming stage
    if stemming_method=='LancasterStemmer':
        if verbose is True: print("\nPerforming Lancaster Stemming method on",input_str)
        stemmer = LancasterStemmer()
    elif stemming_method=='PorterStemmer':
        if verbose is True: print("\nPerforming Porter Stemming method on",input_str)
        stemmer = PorterStemmer()
    else:
        if verbose is True: print('\nNo stemming is done: Output:',input_str)
        input_str = ' '.join(word for word in input_str)
        return input_str
    
    input_str = [stemmer.stem(word) for word in input_str]
    input_str = ' '.join(word for word in input_str)
    
    if verbose is True: print('\nFinal output:',input_str)
    
    return input_str

def thesaurus_matching(iterable_of_iterabls,thesaurus_file='data/thesaurus/thesaurus_for_ai_keyword_with_().csv',have_n_grams=False,verbose=0):
# =============================================================================
#     Matching thesaurus
# =============================================================================
    import pandas as pd
    
    tmp_data = []
    thesaurus = pd.read_csv(thesaurus_file)
    thesaurus = thesaurus.fillna('')
    for row in tqdm(iterable_of_iterabls,total=len(iterable_of_iterabls)):
        doc = []
        for word in row:
            if have_n_grams is False:
#                if thesaurus['alt'].str.match(word).any():
                if word in thesaurus['alt'].values.tolist():
                    if verbose > 0:
                        tmp_word = word
                    word = thesaurus[thesaurus['alt']==word]['original'].values.tolist()[0]
                    if verbose > 0:
                        if tmp_word!=word:
                            print(tmp_word,'->',word)
                doc.append(word)
            else:
                tmp_grams = word.split(' ')
                new_grams = []
                for gram in tmp_grams:
                    if thesaurus[thesaurus['alt'].isin([gram])].shape[0]>0:
                        gram_old = gram
                        gram = thesaurus[thesaurus['alt']==gram]['original'].values.tolist()[0]
                        if verbose > 0:
                            print(gram_old,'->',gram)
                    new_grams.append(gram)
                word = ' '.join(new_grams)
                doc.append(word)
        tmp_data.append(doc)
    return tmp_data

def multiple_replace(string, rep_dict):
    """
    Parameters
    ----------
    string : string
        input string.
    rep_dict : dictionary
        thesaurus dictionary.

    Returns
    -------
    string
     
    
    USAGE
    -------
    multiple_replace("Do you like cafe? No, I prefer tea.", {'cafe':'tea', 'tea':'cafe', 'like':'prefer'})
    >>>> 'Do you prefer tea? No, I prefer cafe.'

    """
    import re
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)

def find_and_remove_c(text):
# =============================================================================
# Text preprocessing 
#    removal of copyright credits from abstract
# =============================================================================
    import re
    
    if text.startswith('Copyright ') or text.startswith('copyright '):
        text = re.sub(r'(Copyright 20).*?.(\.)','',text)
        text = re.sub(r'(Copyright © ).*?.(\.)','',text)
        text = re.sub(r'(copyright 20).*?.(\.)','',text)
        text = re.sub(r'(copyright © ).*?.(\.)','',text)
    else:
        text = re.sub(r'(\(C\)).*?.(\.)','',text)
        text = re.sub(r'(\(c\)).*?.(\.)','',text)
        text = re.sub(r'(Copyright ©).*?.(\.)','',text)
        text = re.sub(r'(Copyright©).*?.(\.)','',text)
        text = re.sub(r'(copyright ©).*?.(\.)','',text)
        text = re.sub(r'(copyright©).*?.(\.)','',text)
        text = re.sub(r'(Copyright ).*?.(\.)','',text)
        text = re.sub(r'(Copyright ).*?.(\.)','',text)
        text = re.sub(r'(©).*?.(\.)','',text)
        text = re.sub(r'(copyright).*?.(\.)','',text)
        text = re.sub(r'(copyright).*?.(\.)','',text)

    
    text = text.replace('all rights reserved','')
    text = text.replace('all rights','')
    text = text.replace('All rights reserved','')
    text = text.replace('All rights','')
    text = text.replace('Rights reserved','')
    text = text.replace('rights reserved','')
    
    return text

def find_and_remove_term(text,from_str,to_str=''):
# =============================================================================
# Text preprocessing 
#    removal of term from text
# =============================================================================
    text = str(text).replace(from_str,to_str)
    return text

def filter_string(text,dictionary):
# =============================================================================
# Text preprocessing 
#   removal/replacing of multiple terms/substrings
# =============================================================================
    for row in dictionary:
        text = str(text).replace(row[0],row[1])
    return text

def get_top_n_words(corpus, n=None):
# =============================================================================
# get to n-grams Credits to https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34
# =============================================================================
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


def sort_coo(coo_matrix):
# =============================================================================
# Sorting tf_idf in descending order Credits to https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34
# =============================================================================
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
# =============================================================================
# Get the feature names and tf-idf score of top n items
# =============================================================================
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results



def keyword_lists_similarity_score(left_list,right_list):    
# =============================================================================
# Compare and score keywords / lists
#    Input:left_list, right_list
#       left_list: main list to compare to. Must have no missing values.
#       right_list: second list to compare. Missing values will be ignored in scoring.
#    Returns: score,count,match,match_score
#       score: comulative score
#       count: total number of the right-side non-none  values
#       match: matched items
#       match_score: number of matches per row
# =============================================================================
    import pandas as pd

    #keywords_keys_df = pd.DataFrame(keywords_keys)
    right_list_df = pd.DataFrame(right_list)
    # Comparison : n-grams
    match = []
    match_score = []
    for i,row in tqdm(right_list_df.iterrows(),total=right_list_df.shape[0]):
        if pd.notnull(row).any():
            intersection = list(set(left_list[i]).intersection(right_list[i]))
            intersection_size = len(intersection)
        else:
            intersection = False
            intersection_size = False
            
        match.append(intersection)
        match_score.append(intersection_size)
        
    score = 0
    count = 0
    for item in match_score :
        if item is not False:
            score = score + item
            count += 1
            
    return score,count,match,match_score


def get_topics(model, count_vectorizer, n_top_words):
# =============================================================================
# Get top n LDA model keywords for all m topics
#    input: model,count_vectorizer,n_top_words
#       model: LDA model trained via scikit-learn
#       count_vectorizer: count vectorizer trained in scikit-learn for corpus
#       n_top_words: number of needed top words (n)
# =============================================================================
    keywords = []
    keyword_scores = []
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        keywords.append([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        keyword_scores.append([topic[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return keywords,keyword_scores



def load_vectors(fname):
# =============================================================================
# Load FastText pre-trained word vectors
#    
#   Pre-Trained Model Credit and Reference:
#    https://fasttext.cc/docs/en/english-vectors.html
#    @inproceedings{mikolov2018advances,
#     title={Advances in Pre-Training Distributed Word Representations},
#     author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
#     booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
#     year={2018} }
# =============================================================================
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def load_vectors_section(fname,from_line=0,to_line=False):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    line_no = 0
    if to_line is False:
        to_line = file_len(fname)
        
    for line in fin:
        if line_no >= from_line and line_no <= to_line:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))
            if line_no % 500 == 0 : print (line_no,'/',to_line)
            if line_no >= to_line:
                return data
        line_no+=1
    return data

def load_vectors_lowmemory_df(fname):
    import pandas as pd
    from tqdm import tqdm

    tqdm.pandas()

    dataset_vector_df = pd.read_csv(fname,delimiter="'-+nothing+-'")
    dataset_vector_df.columns = ['data']
    dataset_vector_df = dataset_vector_df[pd.notnull(dataset_vector_df['data'])]
    dataset_vector_df = dataset_vector_df['data'].str.split(' ',1)
    dataset_vector_df = pd.DataFrame(dataset_vector_df)
    dataset_vector_df['token'] = dataset_vector_df['data'].progress_apply(lambda x: x[0])
    dataset_vector_df['vector'] = dataset_vector_df['data'].progress_apply(lambda x: x[1])
    dataset_vector_df = dataset_vector_df[['token','vector']]
    
    return dataset_vector_df

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3,score_parameter='c_v'):
# =============================================================================
#     Compute c_v coherence for various number of topics
# 
#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics
#     score_parameter : coherence score parameter (c_v, u_mass, c_npmi,...)
#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
# =============================================================================
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.models.ldamodel import LdaModel
    from gensim.corpora.dictionary import Dictionary
    from numpy import array
    
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus,
                       id2word=dictionary,
                       num_topics=num_topics, 
                       random_state=100,
                       update_every=1,
                       chunksize=100,
                       passes=10,
                       alpha='auto',
                       per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=score_parameter)
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def replace_british_american(text, mydict):
    for gb, us in mydict.items():
        text = text.replace(gb, us)
    return text

def get_abstract_keywords(corpus,keywords_wanted,max_df=0.9,max_features=None):
    """
    Parameters
    ----------
    corpus : list
        An iterable list of  of tokens like: ['token1','token2',...]
    keywords_wanted : int
        Number of wanted keywords
    max_df : float, optional
        TF-iDF max_df. The default is 0.9.
    max_features : int, optional
        The max number of vectorizer volcabulary features. The default is None.

    Returns
    -------
    keywords_tfidf : list
        A list of important words for the list of words given.

    """
    cv=CountVectorizer(max_df=max_df,stop_words=stop_words, max_features=max_features, ngram_range=(1,1))
    X=cv.fit_transform(corpus)
    # get feature names
    feature_names=cv.get_feature_names()
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)
    keywords_tfidf = []
    keywords_sorted = []
    for doc in tqdm(corpus,total=len(corpus)):
        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
        sorted_items=kw.sort_coo(tf_idf_vector.tocoo())
        keywords_sorted.append(sorted_items)
        keywords_tfidf.append(kw.extract_topn_from_vector(feature_names,sorted_items,keywords_wanted))
    return keywords_tfidf

def get_corpus_top_keywords(abstract_keywords_dict=None):
    """
    Parameters
    ----------
    abstract_keywords_dict : dictionary
        A dict input in form of tokens and their scores.

    Returns
    -------
    Dataframe
        Dataframe with columns of terms and scores, sorted by scores.

    """
    if abstract_keywords_dict == None:
        print("keywords should be provided")
        return False
    terms = []
    values = []
    for doc in abstract_keywords_dict:
        if doc != None:
            terms = terms+list(doc.keys())
            values = values+list(doc.values())
    terms_df = pd.DataFrame({'terms':terms,'value':values}).groupby('terms').sum().sort_values('value',ascending=False)
    return terms_df
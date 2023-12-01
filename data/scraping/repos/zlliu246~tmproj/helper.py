#Building Question Classifier
# Ref: https://medium.com/analytics-vidhya/naive-bayes-classifier-for-text-classification-556fabaf252b

import pandas as pd
import numpy as np
import re, string, random

#packages
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree

# LDA Model
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import pprint
from gensim.models import CoherenceModel
import spacy

#sklearn & gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

training_data = pd.read_csv("data/Question_Classification_Dataset.csv")
training_data = training_data[["Questions", "Category0"]]
training_data = training_data.rename(columns={"Category0": "class"})

def produce_tdm(df, specific_class):
    D_docs = [row['Questions'] for index,row in training_data.iterrows() if row['class'] == specific_class]
    vec_D = CountVectorizer()
    X_D = vec_D.fit_transform(D_docs)
    tdm_D = pd.DataFrame(X_D.toarray(), columns=vec_D.get_feature_names())

    return tdm_D, vec_D, X_D


tdm_D, vec_D, X_D  = produce_tdm(training_data, "DESCRIPTION")
tdm_E, vec_E, X_E = produce_tdm(training_data, "ENTITY")
tdm_A, vec_A, X_A = produce_tdm(training_data, "ABBREVIATION")
tdm_H, vec_H, X_H = produce_tdm(training_data, "HUMAN")
tdm_N, vec_N, X_N = produce_tdm(training_data, "NUMERIC")
tdm_L, vec_L, X_L = produce_tdm(training_data, "LOCATION")


def produce_freq(vec, X):
    word_list = vec.get_feature_names()
    count_list = X.toarray().sum(axis=0) 
    freq = dict(zip(word_list,count_list))
    freq

    return freq, count_list, word_list


freq_D, count_list_D, word_list_D = produce_freq(vec_D, X_D)
freq_E, count_list_E, word_list_E = produce_freq(vec_E, X_E)
freq_A, count_list_A, word_list_A = produce_freq(vec_A, X_A)
freq_H, count_list_H, word_list_H = produce_freq(vec_H, X_H)
freq_N, count_list_N, word_list_N = produce_freq(vec_N, X_N)
freq_L, count_list_L, word_list_L = produce_freq(vec_L, X_L)


def get_prob(count_list, word_list):
    prob = []
    for count in count_list:
        prob.append(count/len(word_list))
    return dict(zip(word_list, prob))


prob_D = get_prob(count_list_D, word_list_D)
prob_E = get_prob(count_list_E, word_list_E)
prob_A = get_prob(count_list_A, word_list_A)
prob_H = get_prob(count_list_H, word_list_H)
prob_N = get_prob(count_list_N, word_list_N)
prob_L = get_prob(count_list_L, word_list_L)


docs = [row['Questions'] for index,row in training_data.iterrows()]

vec = CountVectorizer()
X = vec.fit_transform(docs)

total_features = len(vec.get_feature_names())

total_cnts_features_D = count_list_D.sum(axis=0)
total_cnts_features_E = count_list_E.sum(axis=0)
total_cnts_features_A = count_list_A.sum(axis=0)
total_cnts_features_H = count_list_H.sum(axis=0)
total_cnts_features_N = count_list_N.sum(axis=0)
total_cnts_features_L = count_list_L.sum(axis=0)

def get_prob_with_qns(new_word_list, freq, total_cnts_features, total_features):
    prob_with_ls = []
    for word in new_word_list:
        if word in freq.keys():
            count = freq[word]
        else:
            count = 0
        prob_with_ls.append((count + 1)/(total_cnts_features + total_features))
    output = dict(zip(new_word_list,prob_with_ls))
    value_list = output.values()
    value_list
    
    prob = 1
    for each in value_list:
        prob *= each
    return prob


def classify_qns(qns):
    new_word_list = word_tokenize(qns)
    
    prob_D = get_prob_with_qns(new_word_list, freq_D, total_cnts_features_D, total_features)
    prob_E = get_prob_with_qns(new_word_list, freq_E, total_cnts_features_E, total_features)
    prob_A = get_prob_with_qns(new_word_list, freq_A, total_cnts_features_A, total_features)
    prob_H = get_prob_with_qns(new_word_list, freq_H, total_cnts_features_H, total_features)
    prob_N = get_prob_with_qns(new_word_list, freq_N, total_cnts_features_N, total_features)
    prob_L = get_prob_with_qns(new_word_list, freq_L, total_cnts_features_L, total_features)

    prob = [prob_D, prob_E, prob_A, prob_H, prob_N, prob_L]
    classes = ["DESCRIPTION", "ENTITY", "ABBREVIATION", 'HUMAN', "NUMERIC", "LOCATION"]
    return(classes[prob.index(max(prob))], max(prob))

#Formulating Query

def get_continuous_chunks(text, label):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
#     print(chunked)
    prev = None
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree and subtree.label() == label:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
#             print('current_chunk', current_chunk)
        if current_chunk:
            named_entity = " ".join(current_chunk)
#             print('named', named_entity)
#             print('continuous', continuous_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk


def formulate_query(qns):
    qns_head = qns.split()[0]
    ner_gpe = get_continuous_chunks(qns, "GPE")
    ner_person = get_continuous_chunks(qns, "PERSON")
    ner_org = get_continuous_chunks(qns, "ORGANIZATION")
    ans_type = classify_qns(qns)
    return [[qns_head], ner_gpe, ner_person, ner_org, ans_type]


#Answer Retrieval by Cosine Similarity

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations and special characters
        
        
def compute_similarity(cleaned_sent_lower):
#     count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer = TfidfVectorizer(stop_words='english')
    sparse_matrix = count_vectorizer.fit_transform(cleaned_sent_lower)
    
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names())
    cosim = cosine_similarity(df, df)
    return cosim

def get_top3(cosim, cleaned_sent_lower):
    threshold = 0.2 #edit this accordingly
    top3prob = np.sort(cosim[-1])[::-1][1:4]
    top3docs = []
    for prob in top3prob:
        if prob >= threshold:
            doc_num = np.where(cosim[-1] == prob)[0][0]
            # print("Doc:", doc_num, ", Cosine:", prob)
#             print(cleaned_sent_lower[doc_num])
            top3docs.append(cleaned_sent_lower[doc_num])
    if top3docs == []:
        top3docs.append("")
    return top3docs

def generate_answer(query_ans_type, qns_head, final_doc):
    #answer full sentence for WHAT, WHY
    sent_tokens = word_tokenize(final_doc)
    tagged_sent = nltk.pos_tag(sent_tokens)
    if final_doc == "":
        return "Sorry, I do not have the answer to this question."    
    elif qns_head[0] == "Who": # Expect name (NNP), of(IN), position(NNP), organization (NNP)
        temp = []
        output = []
        cont = False
        for x,y in tagged_sent:
            if "NNP" in y and cont == False:
                temp.append(x)
                cont = True
            elif "NNP" in y and cont == True:
                output.append(x)
                temp = []
            elif y=="IN" and cont == True:
                output.append(x)
                temp = []
        output = " ".join(output)
        return "The person is " + output #bigram doesnt work
    
    elif qns_head[0] == "Where": # Expect located (VBN) at location (NN)
        output = get_continuous_chunks(final_doc, "GPE")
        return "At " + output[0]
            
    elif qns_head[0] == "When":
        for x,y in tagged_sent:
            if "CD" in y:
                return x
            
    elif query_ans_type == "NUMERIC":
        output = re.findall(r"[$%0-9]+", final_doc) #accept numeric, percentage, price
        return "It is " + output[0]
    
    return final_doc #answer full sentence for WHAT, WHY, HOW as these questions may have a wide variety of paraphrasing

#Evaluate answer

def evaluate_ans_1(query_ans_type, top3docs):
    output = {0:0, 1:0, 2:0}
    if query_ans_type[0] == 'NUMERIC':
        index = 0
        for each in top3docs:
            r1 = re.findall(r"[0-9,]+",each) 
            if r1!=[]:
                output[index] = 1
            index +=1

    elif query_ans_type[0] == 'LOCATION':
        index = 0
        for each in top3docs:
            if get_continuous_chunks(each, "GPE") != []:
                output[index] = 1
            index +=1

    elif query_ans_type[0] == 'HUMAN':
        index = 0
        for each in top3docs:
            if get_continuous_chunks(each, "PERSON") != []:
                output[index] = 1
            index +=1
    return output    


def evaluate_ans_2(query_keywords, top3docs, output):
    for each in query_keywords:
        index = 0
#         print('keywords', query_keywords)
#         print(top3docs)
        for doc in top3docs:
            if each in doc:
                output[index] += 1
            index += 1
    return output

def get_final_doc(top3docs, output):
    max_value = max(output.values())  # maximum value
    max_keys = [k for k, v in output.items() if v == max_value] # getting all keys containing the `maximum`
    return top3docs[max_keys[0]]

#Generate Answer Template

def generate_answer(query_ans_type, qns_head, final_doc):
    #answer full sentence for WHAT, WHY
    sent_tokens = word_tokenize(final_doc)
    tagged_sent = nltk.pos_tag(sent_tokens)
    if final_doc == "":
        return "Sorry, I do not have the answer to this question."    
    elif qns_head[0] == "Who": # Expect name (NNP), of(IN), position(NNP), organization (NNP)
        temp = []
        output = []
        cont = False
        for x,y in tagged_sent:
            if "NNP" in y and cont == False:
                temp.append(x)
                cont = True
            elif "NNP" in y and cont == True:
                output.append(x)
                temp = []
            elif y=="IN" and cont == True:
                output.append(x)
                temp = []
        output = " ".join(output)
        return "The person is " + output #bigram doesnt work
    
    elif qns_head[0] == "Where": # Expect located (VBN) at location (NN)
        output = get_continuous_chunks(final_doc, "GPE")
        return "At " + output[0]
            
    elif qns_head[0] == "When":
        for x,y in tagged_sent:
            if "CD" in y:
                return x
            
    elif query_ans_type == "NUMERIC":
        output = re.findall(r"[$%0-9]+", final_doc) #accept numeric, percentage, price
        return "It is " + output[0]
    
    return final_doc #answer full sentence for WHAT, WHY, HOW as these questions may have a wide variety of paraphrasing


#Answer Question
def answer_question(context, qns):
#     context = str(context.read())
    sentences = context.split(".")

    #qns analysis
    query = formulate_query(qns)
    qns_head = query[0]
    query_keywords = query[1] + query[2] + query[3]
    query_ans_type = query[4]
#     print(query_ans_type)
    
    # Remove trailing \n
    cleaned_sent_lower = [sent.replace("\n", "") for sent in sentences]
#     print("cleaned", cleaned_sent_lower)

    #add test_doc
    cleaned_sent_lower.append(qns)

    #compute similarity
    cosim = compute_similarity(cleaned_sent_lower)
    
    #gettop3 docs
    top3docs = get_top3(cosim, cleaned_sent_lower)
#     print(top3docs)

    # get evaluated ans I
    output = evaluate_ans_1(query_ans_type, top3docs)
#     print(output)
    
     # get evaluated ans II
    output = evaluate_ans_2(query_keywords, top3docs, output)
#     print(output)
    
    #get ans
    final_doc = get_final_doc(top3docs, output)
    
    #answer template
    return generate_answer(query_ans_type, qns_head, final_doc)
import os
import re
import spacy
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2000000
from spacy.lang.en import English
parser = English()

from gensim.utils import simple_preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


f = open("Results_all/cpp/02_commit.txt",encoding="utf-8-sig", errors='ignore')

test_str = f.readlines()
test_str = [re.sub('\S*@\S*\s?', '', sent) for sent in test_str]
test_str = [re.sub('\s+', ' ', sent) for sent in test_str]
test_str = [re.sub('\d','',sent) for sent in test_str]
test_str = [re.sub('[,\.!?]','',sent) for sent in test_str]
test_str = [sent.lower() for sent in test_str]


# Remove distracting single quotes
res_first = test_str[0:len(test_str)//2] 
res_second = test_str[len(test_str)//2 if len(test_str)%2 == 0
                                else ((len(test_str)//2)+1):] 


# doc = nlp(''.join(ch for ch in f.read() if ch.isalnum() or ch == " "))
# print(doc)
data = []

data.append(res_first)
data.append(res_second)

import gensim
from gensim.utils import simple_preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))
# print(data_words[:1])


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)




# NLTK Stop words
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out



import spacy
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:1])    

from gensim.models import CoherenceModel

import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
# print(corpus[:1])

if __name__ == "__main__":

    tFile = "Mtopics.txt"
    topics_file = open(tFile,'w',encoding="utf-8-sig", errors='ignore')
    coherence = "coh.txt"
    coh_file = open(coherence,'w',encoding="utf-8-sig", errors='ignore')
# Build LDA model
    for n in range(2,10):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=n, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)

        
        lda_model.save('model'+str(n)+'.gensim')
        topics = lda_model.print_topics(num_words=10)
        topics_file.write("NUMTopics = "+str(n)+"\n")
        for topic in topics:
            print(topic)
            topics_file.write(str(topic)+"\n")
        topics_file.write("\n------------------------\n") 

        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        coh_file.write('\nTopic: '+str(n)+'----Coherence Score :'+str(coherence_lda)+"\n-----------------------------------") 
        print(n)  

   
        

# import numpy as np
# import tqdm
# import pandas as pd
# from pprint import pprint
# # # Print the Keyword in the 10 topics
# # # pprint(lda_model.print_topics())
# # doc_lda = lda_model[corpus]
# from gensim.models import CoherenceModel
# from gensim.models import LdaMulticore
# # # Compute Coherence Score
# if __name__ == "__main__":
#     # freeze_support()

#     coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
#     coherence_lda = coherence_model_lda.get_coherence()
#     print('\nCoherence Score: ', coherence_lda)
# #     # supporting function
# #     def compute_coherence_values(corpus, dictionary, k, a, b):
        
# #         lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
# #                                                id2word=id2word,
# #                                                num_topics=10, 
# #                                                random_state=100,
# #                                                chunksize=100,
# #                                                passes=10,
# #                                                alpha=a,
# #                                                eta=b,
# #                                                per_word_topics=True)
        
# #         coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        
# #         return coherence_model_lda.get_coherence()

# #     grid = {}
# #     grid['Validation_Set'] = {}
# #     # Topics range
# #     min_topics = 2
# #     max_topics = 11
# #     step_size = 1
# #     topics_range = range(min_topics, max_topics, step_size)
# #     # Alpha parameter
# #     alpha = list(np.arange(0.01, 1, 0.3))
# #     alpha.append('symmetric')
# #     alpha.append('asymmetric')
# #     # Beta parameter
# #     beta = list(np.arange(0.01, 1, 0.3))
# #     beta.append('symmetric')
# #     # Validation sets
# #     num_of_docs = len(corpus)
# #     print("len:"+str(len(corpus)))
# #     corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
# #                    # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
# #                    gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
# #                    corpus]

# #     # print("len of corpus"+str(len(corpus_sets)))               
# #     corpus_title = ['75% Corpus', '100% Corpus']
# #     model_results = {'Validation_Set': [],
# #                      'Topics': [],
# #                      'Alpha': [],
# #                      'Beta': [],
# #                      'Coherence': []
# #                     }
# #     # Can take a long time to run
# #     if 1 == 1:
# #         pbar = tqdm.tqdm(total=50)
        
# #         # iterate through validation corpuses
# #         for i in range(len(corpus_sets)):
# #             # iterate through number of topics
# #             for k in topics_range:
# #                 # iterate through alpha values
# #                 for a in alpha:
# #                     # iterare through beta values
# #                     for b in beta:
# #                         # get the coherence score for the given parameters
# #                         cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
# #                                                       k=k, a=a, b=b)
# #                         # Save the model results
# #                         model_results['Validation_Set'].append(corpus_title[i])
# #                         model_results['Topics'].append(k)
# #                         model_results['Alpha'].append(a)
# #                         model_results['Beta'].append(b)
# #                         model_results['Coherence'].append(cv)
                        
# #                         pbar.update(1)
# #         pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
# #         pbar.close()

#         lda_model = gensim.models.LdaMulticore(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=5, 
#                                            random_state=100,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha=0.01,
#                                            eta=0.61)

#         from pprint import pprint
#         # Print the Keyword in the 10 topics
#         pprint(lda_model.print_topics())
#         doc_lda = lda_model[corpus]


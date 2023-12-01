import csv
import os
import re
import spacy
from pprint import pprint
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2000000
from spacy.lang.en import English
parser = English()

from gensim.utils import simple_preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

with open('resultcsharp_comment.csv', 'w', newline='',encoding="utf-8-sig", errors='ignore') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(["Name", "T0","T1","T2","T3","T4","Topics"])


    for root, dirs, files in os.walk(('./csharp')):
        for file in files:
            # print(file)
            if(file.endswith("comment.txt")):
                fname = (os.path.join(root, file))


                f = open(fname,encoding="utf-8-sig", errors='ignore')

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
                data_words_1 = list(sent_to_words(data[0]))
                data_words_2 = list(sent_to_words(data[1]))
                data_words = data_words_1 + data_words_2
                print(len(data_words))
                # Build the bigram and trigram models
                if(len(data_words)>1):

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
                    import pandas as pd
                    # Remove Stop Words
                    data_words_nostops = remove_stopwords(data_words)
                    # Form Bigrams
                    data_words_bigrams = make_bigrams(data_words_nostops)
                    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
                    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
                    # Do lemmatization keeping only noun, adj, vb, adv
                    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
                    # print(data_lemmatized[:1])    

                    # from gensim.models import CoherenceModel

                    import gensim.corpora as corpora
                    # Create Dictionary
                    id2word = corpora.Dictionary(data_lemmatized)
                    # Create Corpus
                    texts = data_lemmatized
                    # Term Document Frequency
                    corpus = [id2word.doc2bow(text) for text in texts]
                    # View
                    # print(id2word[0])
                    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                               id2word=id2word,
                                                               num_topics=5, 
                                                               random_state=100,
                                                               update_every=1,
                                                               chunksize=100,
                                                               passes=10,
                                                               alpha='auto',
                                                               per_word_topics=True)

                    # pprint(lda_model.print_topics())
                    doc_lda = lda_model[corpus]
                    # Compute Perplexity
                    optimal_model = lda_model
                    model_topics = optimal_model.show_topics(formatted=False)
                    get_document_topics = lda_model.get_document_topics(corpus)
                    # print(get_document_topics[1])
                    import operator
                    mylist = []
                    for i in get_document_topics:
                        # se = []
                        i.sort(key = operator.itemgetter(1), reverse = True)
                        mylist.append(i[0][0])
                    c0 = mylist.count(0)
                    c1 = mylist.count(1)
                    c2 = mylist.count(2)
                    c3 = mylist.count(3)
                    c4 = mylist.count(4)

                    t = len(mylist)

                    t0 = c0/t * 100       
                    t1 = c1/t * 100
                    t2 = c2/t * 100
                    t3 = c3/t * 100
                    t4 = c4/t * 100
                    # pprint(lda_model.print_topics())
                    topics = lda_model.print_topics(num_words=10)
                    # topics_file.write("NUMTopics = "+str(n)+"\n")
                    
                    writer.writerow([file,t0,t1,t2,t3,t4,topics])


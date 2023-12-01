import matplotlib.pyplot as plt
import nltk
stopwords = nltk.corpus.stopwords.words('english')
import numpy as np
import operator
import pandas as pd
import re

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf



def identity_tokenizer(text):
    return text



def pos_tagger(nltk_tag): 
    """
    """
    if nltk_tag.startswith('J'): 
        return nltk.corpus.wordnet.ADJ 
    elif nltk_tag.startswith('V'): 
        return nltk.corpus.wordnet.VERB 
    elif nltk_tag.startswith('N'): 
        return nltk.corpus.wordnet.NOUN 
    elif nltk_tag.startswith('R'): 
        return nltk.corpus.wordnet.ADV 
    else:           
        return None



def abstract_to_BagofWords(text, stopwords=stopwords):
    """
    """
    ### tokenizing
    tokens = nltk.word_tokenize(text)

    ### remove punctuation
    punc = [',', '.', ';', '!', '?', '(', ')', ':', '-', "'", '"', "'s", 'the']
    filtered1 = [i for i in tokens if i not in punc]

    ### remove numbers
    filtered2 = [re.sub('[0-9]+\.*[0-9]+', '', i) for i in filtered1]
    filtered2 = [i for i in filtered2 if i != '']

    ### part-of-speech tagging
    tagged = nltk.pos_tag(filtered2)

    ### lemmatizing
    lemmatizer = nltk.stem.WordNetLemmatizer()
    processed = []
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tagged))  # simplify PoS tags to lemmatize
    for word, tag in wordnet_tagged:
        if tag is None:
            processed.append(word.lower())
        else:
            tmp = lemmatizer.lemmatize(word.lower(), tag)
            processed.append(tmp)

    ### remove single letter words
    processed = [word for word in processed if len(word) > 1]

    ### remove stopwords
    processed = [i for i in processed if i not in stopwords]

    return list(processed)



def best_no_of_topics(tokenized_texts, range=(5,75), step=5, visualize=False):
    """
    """
    # map words to their integer id
    dictionary = Dictionary(tokenized_texts)

    # filter extremes 
    dictionary.filter_extremes(no_below=3, no_above=0.95)

    # bag-of-words format
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    topic_nums = list(np.arange(range[0], range[1], step))

    # Run a NMF model and calculate the coherence score for each number of topics
    coherence_scores = []
    for num in topic_nums:
        nmf = Nmf(corpus=corpus, num_topics=num, id2word=dictionary, chunksize=2000, passes=5,
            kappa=.1, minimum_probability=0.01, w_max_iter=300, w_stop_condition=0.0001, h_max_iter=100,
            h_stop_condition=0.001, eval_every=10, normalize=True, random_state=42)
        
        cm = CoherenceModel(model=nmf, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(round(cm.get_coherence(), 5))

    # number of topics with the highest coherence score
    scores = list(zip(topic_nums, coherence_scores))
    best_num_topics = sorted(scores, key=operator.itemgetter(1), reverse=True)[0][0]  

    #visualize
    if visualize == True:
        fig = plt.figure(figsize=(15, 7))
        plt.plot(topic_nums, coherence_scores, linewidth=3, color='#4287f5')

        plt.xlabel("Number of topics", fontsize=14)
        plt.ylabel("Coherence Score", fontsize=14)
        plt.title('Coherence Score by Topic Number - Best Number of Topics: {}'.format(best_num_topics), fontsize=18)
        plt.xticks(np.arange(5, max(topic_nums) + 1, 5), fontsize=12)
        plt.yticks(fontsize=12)

        fig.savefig('../deliverables/coherence_score.png', dpi=fig.dpi, bbox_inches='tight')
        plt.show()
    return best_num_topics



def topic_table(nmf_model, vectorizer, n_top_words=8):
    topics = {}
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(nmf_model.components_):
        t = (topic_idx)
        topics[t] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1: -1]]
    return pd.DataFrame(topics)



def predict_topic(nmf_model, vectorizer, abstracts):
    if type(abstracts) == str:
        abstracts = [abstracts]

    new_BoWs= [] # NLP preprocessed bag of words
    for i in abstracts:
        tmp = abstract_to_BagofWords(i, stopwords=stopwords)
        new_BoWs.append(tmp)

    ### Make predictions for new papers
    features_new = tfidf.transform(new_BoWs)
    X_new = nmf.transform(features_new)

    # Get the top predicted topics
    return [np.argsort(each)[::-1][0] for each in X_new]
    
import heapq
from msilib.schema import Error
import pandas as pd
from gensim.models import CoherenceModel
from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance


def wordFreqGenerator (tokens):
    wordfreq = {}
    for sentence in tokens:
        for token in sentence:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    #get the words with the highest freq
    most_freq = heapq.nlargest(10, wordfreq, key=wordfreq.get)
    print (f"Top 20 words: {most_freq}")

    return wordfreq

def TFIDFGenerator (docs):
    #TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfIdfVectorizer = TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform([" ". join(i) for i in docs])
    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)

    print ("Top words with the highest TFIDF scores")
    print (df.head(25))
    print (df.describe())

    return df

# supporting function
def computeCoherenceValues(tokens, id2word, model=None, topics=None, process_num=-1):
    process_param = process_num
    window_size = 48
    top_words_num = 10
        
    if model != None:
        cv_coherence = CoherenceModel(model=model, texts=tokens, dictionary=id2word, coherence='c_v', window_size=window_size, topn=top_words_num, processes=process_param)
        cuci_coherence = CoherenceModel(model=model, texts=tokens, dictionary=id2word, coherence='c_uci', window_size=window_size, topn=top_words_num, processes=process_param)
    else:
        topics = list(topics.values())
        try:
            cv_coherence = CoherenceModel(topics = topics, texts=tokens, dictionary=id2word, coherence='c_v', window_size=window_size, topn=top_words_num, processes=process_param) 
            cuci_coherence = CoherenceModel(topics = topics, texts=tokens, dictionary=id2word, coherence='c_uci', window_size=window_size, topn=top_words_num, processes=process_param) 
            
            # print("Success")
            # print (len(topics))
            # print(topics)
        except ValueError:
            # print("Failure")
            # print (len(topics))
            # print (topics)
            raise BaseException()

    return cv_coherence.get_coherence(), cuci_coherence.get_coherence()

def cluster (tokens, embeddings, num_of_topics):
    #kclusterer = KMeansClusterer(num_of_topics, distance=cosine_distance, repeats=25, avoid_empty_clusters=True)
    from sklearn.cluster import KMeans

    kclusterer = KMeans(
                        n_clusters=num_of_topics, init='random',
                        n_init=10, max_iter=300, 
                        tol=1e-04, random_state=0)
    assigned_clusters = kclusterer.fit_predict(embeddings)

    clusters = {}
    for i, token in enumerate(tokens): 
        if assigned_clusters[i] in clusters.keys():
            clusters[assigned_clusters[i]].append(token)
        else:
            clusters[assigned_clusters[i]] = [token]

    return clusters

def cluster_with_ctfidf (topic_clusters):
    flattened_clusters = {}

    for key in sorted(topic_clusters.keys()):
        doc_in_cluster = []

        for doc in topic_clusters[key]:
            doc_in_cluster.extend(doc)
        
        flattened_clusters[key] = " ".join(doc_in_cluster)
    
    from sklearn.feature_extraction.text import CountVectorizer
    from ctfidf import CTFIDFVectorizer

    # Create bag of words
    count_vectorizer = CountVectorizer().fit(flattened_clusters.values())
    count = count_vectorizer.transform(flattened_clusters.values())
    words = count_vectorizer.get_feature_names()

    # Extract top 10 words
    ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(flattened_clusters)).toarray()
    
    words_per_topic = {}
    for key in flattened_clusters.keys():
        try:
            key_words = ctfidf[key]
        except IndexError:
            print ([len(words) for words in flattened_clusters.values()])
            print (f'Flattened clusters: {flattened_clusters.keys()}')
            print (f'Array shape: {ctfidf.shape}')
            pass
        sorted_key_words = key_words.argsort()[-10:]
        words_per_topic[key] = [words[index] for index in sorted_key_words]

    return words_per_topic

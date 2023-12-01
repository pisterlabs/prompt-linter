from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def get_topics_coherence(topics, corpus, id2word):
    cm = CoherenceModel(topics=topics, texts=corpus, dictionary=id2word, coherence='c_v') # note that a dictionary has to be provided.
    return cm.get_coherence()

if __name__ == "__main__":
    import csv

    encoding = "utf-8"
    word2veccluster2words = dict()
    GloVe2words = dict()
    with open("data/category_trainedWord2vec-KMeans-Food_word2cluster.csv", 'r') as f:
        reader = csv.reader(f)
        skip = True
        for row in reader:
            if skip:
                skip = False
            else:
                if row[1] not in word2veccluster2words:
                    word2veccluster2words[row[1]] = []
                word2veccluster2words[row[1]].append(row[0])

    with open("data/GloVe-KMeans-Food_word2cluster.csv", 'r') as f:
        reader = csv.reader(f)
        skip = True
        for row in reader:
            if skip:
                skip = False
            else:
                if row[1] not in GloVe2words:
                    GloVe2words[row[1]] = []
                GloVe2words[row[1]].append(row[0])

    import pickle
    category = "Food"
    with open("data/Las_Vegas-" + category + ".p", "rb") as f:
        all_reviews, business_id2business, user_id2user = pickle.load(f)

    id2word = dict()
    corpus  = []
    for review in all_reviews:
        corpus.append(review["words"])
        for word in review["words"]:
            if word not in id2word:
                id2word[len(id2word)] = word

    id2word = Dictionary(corpus)
    corpus = [id2word.doc2bow(text) for text in corpus]
    topics = list(word2veccluster2words.values())
    print("word2vec",get_topics_coherence(topics, corpus, id2word))
    topics = list(GloVe2words.values())
    print("GloVe",get_topics_coherence(topics, corpus, id2word))

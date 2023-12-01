import os
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt


def get_data():
        dir_name = os.path.join(".","data2000")
        with open(os.path.join(dir_name, "documents.txt"), "r") as documents_file:
            data = documents_file.read().splitlines()
            documents = [[int(w) for w in d.split(' ')] for d in data if d != '']
        with open(os.path.join(dir_name, "vocab.txt"), "r") as vocab_file:
            vocab = vocab_file.read().split(' ')
        return vocab,documents

def pure_lda():
    vocab,docs = get_data()
    list_of_docs = []
    for doc in docs:
        doc_str_list = []
        for word_id in doc:
            word = vocab[word_id]
            doc_str_list.append(word)
        list_of_docs.append(doc_str_list)
# Create Dictionary 
    id2word = corpora.Dictionary(list_of_docs)
# Create Corpus 
    texts = list_of_docs
# Term Document Frequency 
    corpus = [id2word.doc2bow(text) for text in texts]  

#     lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=20, 
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)

#     for i, topic in lda_model.show_topics(num_topics=10, formatted=False):
#         print('Topic {}: {}'.format(i, ', '.join([word[0] for word in topic])))
# # Compute Perplexity
#     print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
# a measure of how good the model is. lower the better.

# Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda_model,topn=1, texts=list_of_docs, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence_per_topic()
    # print('\nCoherence Score: ', coherence_lda)
    coherence_values = []
    num_topics_list = []
    for num_topics in range(1, 22, 5):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        num_topics_list.append(f"{num_topics}")
        coherencemodel = CoherenceModel(model=model,topn=10, texts=list_of_docs, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return num_topics_list, coherence_values
def plot(x,y,type):
    # Show graph
    plt.figure(figsize=(8,6))
    plt.plot(x, y,label=type)
    plt.title(f"Coherence score/ num topics ({type})")
    plt.xlabel("n_topics")
    plt.ylabel("Coherence score")
    plt.legend()
    plt.xticks(rotation=45)

    plt.savefig(f"{type}_n_topics_news_pure.png")
if __name__ == "__main__":
    x,y = pure_lda()
    pprint(x)
    pprint(y)
    plot(x,y,'c_v')
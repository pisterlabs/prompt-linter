import re, numpy as np, pandas as pd
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import os.path
import pyLDAvis
import pyLDAvis.gensim_models
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
import fr_core_news_md
import config 

##charge spacynlp
nlp = fr_core_news_md.load()
number_of_topics=4
stop_words = list(stopwords.words('french'))+['ctre','tcdicount']
#stop_words =stop_words +

##lemmatize word with spacy#############
def lemma_word(word):
    doc=nlp(word)
    for t in doc:
        token=t.lemma_
    return token
###prerocess#####
def preprocess(row):
    content = row['text']
    tokens = nltk.word_tokenize(content)
    tokens= [w for w in tokens if w.isalpha()]
    tokens= [lemma_word(word) for word in tokens]
    tokens=[word for word in tokens if len(word)>3]
    tokens= [word for word in tokens if not word in stop_words]
    #print(tokens)
    return tokens
###### preparecorpus#####
def prepare_corpus(doc_clean):
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    return dictionary,doc_term_matrix
####LSA #############
def create_gensim_lda_model(doc_clean,number_of_topics,words):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    #print(dictionary,doc_term_matrix)
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=number_of_topics,id2word = dictionary,random_state=100,update_every=1,chunksize=100,passes=10,alpha='auto', per_word_topics=True)  # train model
    #print(ldamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return ldamodel
####Compute coherence####
def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=1, step=2):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LDA model
        model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values
### plot graph
def plot_graph(doc_clean,start, stop, step):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig('coherence.png')
    #plt.show()

















if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    print(df.head())
    print(df.shape)
    df=df.rename(columns={"date_published": "date"})
    column=['Unnamed: 0']
    df = df.drop(column, axis=1)
    df['text'] = df['body']
    df.date = pd.to_datetime(df.date)
    timezone_name = 'Europe/Paris'
    df.date=df.date.dt.tz_convert(timezone_name)
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('[^\w\s]','')
    df['words'] = df.apply(preprocess, axis=1)
    df=df[['text', 'words']]
    clean_text=list(df.words)
    words=12
    model=create_gensim_lda_model(clean_text,number_of_topics,words)
    print(model)
    start,stop,step=2,8,1
    plot_graph(clean_text,start,stop,step)
    dictionary,doc_term_matrix=prepare_corpus(clean_text)
    vis = pyLDAvis.gensim_models.prepare(model, doc_term_matrix, dictionary)
    pyLDAvis.save_html(vis, 'lda.html')
    




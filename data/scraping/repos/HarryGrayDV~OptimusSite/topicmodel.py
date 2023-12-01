import csv
import pandas as pd
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

PATH = 'nlp/inputs.csv'
def topic_matcher(PATH,other_texts)
    # Import and read the csv file
    ifile  = open(PATH, "r")
    read = csv.reader(ifile)
    doc_complete = []
    for row in read :
        doc_complete.append(row[0])

    # Remove stopwords, punctuation and normalize
    stopwords = stopwords.words('english')
    stopwords.append('lacroix')
    stop = set(stopwords)

    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]  

    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)
    topics = ldamodel.print_topics(num_words=3)
    for topic in topics:
        print(topic)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: {:0.2f}%'.format(coherence_lda*100))

    # Create a new corpus, made of previously unseen documents.
    #other_texts = ['naturally water', 'just buy', 'things', 'sparkling feeling']
    clean_other = [clean(doc).split() for doc in other_texts]
    other_corpus = [dictionary.doc2bow(doc) for doc in clean_other]

    call_to_action_df = pd.DataFrame(columns = ['t0', 't1','t2', 't3', 't4'], index = other_texts)
    call_to_action_df['t0'] = [0]*len(other_texts)
    call_to_action_df['t1'] = [0]*len(other_texts)
    call_to_action_df['t2'] = [0]*len(other_texts)
    call_to_action_df['t3'] = [0]*len(other_texts)

    for doc in range(len(other_corpus)):
        unseen_doc = other_corpus[doc]
        vector = ldamodel[unseen_doc] # get topic probability

        call_to_action_df.iloc[doc,0] = round(vector[0][1],4)
        call_to_action_df.iloc[doc,1] = round(vector[1][1],4)
        call_to_action_df.iloc[doc,2] = round(vector[2][1],4)
        call_to_action_df.iloc[doc,3] = round(vector[3][1],4)
        call_to_action_df.iloc[doc,4] = round(vector[4][1],4)

    return call_to_action_df
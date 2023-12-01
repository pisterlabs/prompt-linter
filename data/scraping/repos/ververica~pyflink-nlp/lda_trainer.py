import pandas as pd
import spacy
from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess as spp
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.test.utils import datapath

"""
    Latent Dirichlet Allocation (LDA) is a bag-of-words algorithm that allows 
    to automatically discover topics contained within a given set of documents.

    It is more flexible than K-Means in the sense that documents can belong to 
    multiple clusters.

    You can find sample data to play with and the original training set at: 
    https://drive.google.com/file/d/1ugnWEQV19g0qTDz0LJkjr8VgujkAb2ez/view?usp=sharing
"""

def lemmatize(tl_bg, allowed_postags=['NOUN','VERB','ADJ','ADV']):
    tl_out = []
    for tkn in tl_bg:
        doc = nlp(" ".join(tkn))
        tl_out.append([t.lemma_ for t in doc if t.pos_ in allowed_postags])

    return tl_out


def clean(tl):

    tl_stopwords = [[word for word in spp(str(tkn)) if word not in STOPWORDS] for tkn in tl]

    bigram = Phrases(tl_stopwords, min_count=5, threshold=100)

    bigram_mod = Phraser(bigram)

    tl_bigram_model = [bigram_mod[tkn] for tkn in tl_stopwords]

    out = lemmatize(tl_bigram_model)

    return out


def tokenize(msg):
    for m in msg:
        yield (spp(str(m), deacc=True))


def format_topics_sentences(ldamodel, corpus, texts):

    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the dominant topic, percentage of contribution and keywords
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # Dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


if __name__ == '__main__':

    df = pd.read_csv("<path-to-dir>/<training-set>.csv")

    token_list = list(tokenize(df["message_subject"]))

    nlp = spacy.load('en', disable=['parser', 'ner'])

    texts = clean(token_list)

    # Build a index to word Dictionary
    id2word = Dictionary(texts)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=22,
                         random_state=100,
                         update_every=1,
                         chunksize=1000,
                         passes=10,
                         alpha='auto')

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, texts)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    print(df_dominant_topic.groupby(['Dominant_Topic']).size())

    # Save the model

    temp_file = datapath("<path-to-dir>/lda_model/lda_model_user_ml")
    lda_model.save(temp_file)

    # print(lda_model.print_topics(num_topics=24,num_words=5))
    # print('\nPerplexity: ', lda_model.log_perplexity(corpus))
    # print('\nCoherence Score: ', coherence_lda)
    # print('Number of unique tokens: %d' % len(id2word))
    # print('Number of documents: %d' % len(corpus))

import pandas as pd
import numpy as np
import jieba
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from datetime import datetime
from .common import ModelCommonUse

def Run(filename):
    data_path = "/home/csi/project_NLP/data/"
    filepath = data_path + filename
    # 需要對沒有偵測到意圖的分類
    collect_df = pd.read_csv(filepath)

    other_df = collect_df[(collect_df["意圖"].isna())]
    other_df = other_df[~other_df["內容"].isna()]
    mapping_other_df = other_df.copy()
    other_df = other_df[other_df["動作"]!="打分"]

    def stopwordslist(filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords

    def remove_stopwords(data_words):
        return [[word for word in doc if word not in stop_words] for doc in data_words]

    # 斷詞
    def sent_to_words(sentences):
        jieba.load_userdict("./data/dictword.txt")
        for sentence in sentences:
            yield(jieba.lcut(str(sentence)))

    data = other_df["內容"].values.tolist()
    data_words = list(sent_to_words(data))

    stop_words = stopwordslist("./data/stopwords.txt")
    data_words_stop = remove_stopwords(data_words)

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    data_words_bigrams = make_bigrams(data_words_stop)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words_bigrams)
    # Create Corpus
    texts = data_words_bigrams
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Build LDA model
    # for i in range(1,21):
    i = 15
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=i, 
                                        random_state=100,
                                        chunksize=100,
                                        passes=10,
                                        per_word_topics=True)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)

    def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = row[0]
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)


    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

    # Format
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', '內容']

    df_result = df_dominant_topic[["Dominant_Topic", "Keywords", "內容"]]
    df_result.drop_duplicates(subset="內容", inplace=True)
    final_df = mapping_other_df.join(df_result.set_index(["內容"]), on=["內容"])
    # put Out of Intent json file path below
    # OOI_path = 
    final_df.to_csv("./data/ouput/Out_of_Intent" + str(timestamp) + ".csv", encoding="utf-8-sig", index=False)
    result = ModelCommonUse().outputjson(final_df, "Dominant_Topic", filepath)
    return result
import pandas as pd
import sqlite3
import time
import nltk
import spacy
import nl_core_news_sm
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis.gensim
mallet_path = 'C:\\Users\Paul\Downloads\mallet-2.0.8\mallet-2.0.8\\bin\mallet'


def create_database_connection() -> bool:
    try:
        global conn, c
        conn = sqlite3.connect(sqlite_file)
        c = conn.cursor()
        print("Connected to database.")
        return True
    except sqlite3.Error as e:
        print(e)
        return False


def create_bigram(data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=80)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in data_words]


def save_noun(doc_clean, language):
    if language == 'en':
        new_doc = []
        for doc in doc_clean:
            tagged = nltk.pos_tag(doc)
            for (word, tag) in tagged:
                if not tag.startswith("N"):
                    doc.remove(word)
                else:
                    continue
            new_doc.append(doc)
        return new_doc
    elif language == 'nl':
        nlp = nl_core_news_sm.load()
        new_doc = []
        for doc in doc_clean:
            doc_for_tagging = nlp(doc)
            for word in doc_for_tagging:
                if str(word) in ["wer", "wel", "mooi", "onz", "gan", "mak", "waarschijn", "leuk", "hel"]:
                    doc = doc.replace(str(word) + " ", "")
                else:
                    continue
                tag = word.pos_
                if not tag == "NOUN":
                    doc = doc.replace(str(word) + " ", "")
                else:
                    continue
            new_doc.append(doc)
        return new_doc


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, passes=10)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


start_time = time.time()

# Path to database
sqlite_file = 'C:\\Users\Paul\Desktop\db_tweets_27Jan.sqlite'
create_database_connection()

# Language of tweets we look at
language = 'nl'
print("Looking at %s tweets." % language)

tweets_grouped = pd.read_sql_query("SELECT DISTINCT hashtag.text, group_concat(cleaned_tweets.stem_tweets, ' ') "
                                   "FROM cleaned_tweets inner join hashtag on cleaned_tweets.id = hashtag.tweet_id "
                                   "WHERE (cleaned_tweets.stem_tweets is not NULL "
                                   "AND cleaned_tweets.clean_retweets is NULL) "
                                   "AND tweet_id in (SELECT id FROM tweets WHERE (lang='%s')) "
                                   "group by hashtag.text" % language, conn)

tweets_grouped = tweets_grouped.rename(columns={"group_concat(cleaned_tweets.stem_tweets, ' ')": "clean_tweets"})

# Saving only Nouns
print("Saving only nouns")
tweets_list = tweets_grouped["clean_tweets"].tolist()
tweets_list_noun = save_noun(tweets_list, 'nl')
words_list_nouns = list(sent_to_words(tweets_list_noun))


# Looking for combinations of two words that occur together often
doc_for_lda = create_bigram(words_list_nouns)

# Create dictionary
print('Create Dictionary')
id2word = corpora.Dictionary(doc_for_lda)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
print('Converting list of documents')
corpus = [id2word.doc2bow(doc) for doc in doc_for_lda]

# Create the object for LDA model
print('Create object for LDA')
Lda = gensim.models.ldamodel.LdaModel

"""
# Find optimal amount of topics
limit = 80
start = 2
step = 2
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=doc_noun,
                                                        start=start, limit=limit, step=step)
# Plot results
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend("coherence_values", loc='best')
plt.show()
"""

# Running and Training LDA model on the document term matrix.
print('Train LDA')
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=5, passes=10)

print(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity and Coherence Score
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_for_lda, corpus=corpus,
                                     dictionary=id2word, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics and save in html file
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization_nl_nouns3.html')

print("--- %s seconds ---" % (time.time() - start_time))
conn.close()

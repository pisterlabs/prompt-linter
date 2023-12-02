import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import re

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Function to compute coherence score
def compute_coherence_score(dictionary, corpus, texts, start, limit, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit + 1, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=0)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

# Function to display topics from LDA model
def display_optimal_lda_topics(model, dictionary, no_top_words):
    for topic_id, topic in model.show_topics(num_topics=model.num_topics, num_words=no_top_words, formatted=False):
        print(f"Topic {topic_id}:")
        topic_words = [word for word, _ in topic]
        print(" ".join(topic_words))

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

    file = 'by_newest_20231113_000949_1st_translated'
    file_path = f'../../gmr_scraper/translated_reviews/{file}.csv'
    reviews_df = pd.read_csv(file_path)

    processed_texts = [preprocess_text(text) for text in reviews_df['Translated_Review'].dropna()]
    word_freq = Counter(word for text in processed_texts for word in text)
    potential_stopwords = [word for word, freq in word_freq.most_common(25)]
    custom_stopwords = potential_stopwords
    stop_words = set(stopwords.words('english')).union(custom_stopwords)
    processed_texts = [[word for word in text if word not in stop_words] for text in processed_texts]
    processed_texts_str = [" ".join(text) for text in processed_texts]
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(processed_texts_str)

    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    start = 2
    limit = 20
    step = 1
    model_list, coherence_values = compute_coherence_score(dictionary=dictionary, corpus=corpus, texts=processed_texts, start=start, limit=limit, step=step)
    optimal_idx = np.argmax(coherence_values)
    optimal_num_topics = start + optimal_idx * step

    print(f"Optimal Number of Topics: {optimal_num_topics}, Coherence Score: {coherence_values[optimal_idx]}")

    optimal_lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_num_topics, random_state=0)
    no_top_words = 10
    display_optimal_lda_topics(optimal_lda, dictionary, no_top_words)
    
    coherence_value_formatted = format(coherence_values[optimal_idx], '.3f')

    file_base_name = f"{coherence_value_formatted}_topic_modeling_results_{file}"

    topics_df = pd.DataFrame({
        'Topic_ID': range(optimal_lda.num_topics),
        'Top_Words': [", ".join([word for word, _ in optimal_lda.show_topic(i, topn=no_top_words)]) for i in range(optimal_lda.num_topics)]
    })

    topics_csv_filename = f"{file_base_name}.csv"
    topics_df.to_csv(topics_csv_filename, index=False)
    
    model_filename = f"{file_base_name}_lda_model"
    optimal_lda.save(model_filename)

    dictionary_filename = f"{file_base_name}_dictionary.gensim"
    dictionary.save(dictionary_filename)

    corpus_filename = f"{file_base_name}_corpus.mm"
    corpora.MmCorpus.serialize(corpus_filename, corpus)
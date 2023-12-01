import pandas as pd
import re
import string
import time
import nltk
from gensim import corpora
from gensim.models import CoherenceModel
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic

DATA_PATH = '/content/pubmedgenerale.csv'
RE_URL = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
RE_EMAIL = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
RE_REPETITION = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(path):
    return pd.read_csv(path, delimiter=',', error_bad_lines=False)

def cleaning(text):
    #Transform to lower-case
    text = text.lower()
    #Remove urls
    text = re.sub(re_url, ' ', text)
    #Remove email addresses
    text = re.sub(re_email, ' ', text)
    #Remove square brackets
    text = re.sub('\[.*?\]', '', text)
    #Remove \n and \r
    text = text.replace('\\n', ' ').replace('\\r', ' ')
    #Remove digit
    text = re.sub(r'(\d+)', ' ', text)
    #Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    #Remove emojis
    #text= remove_emoji(text)
    #Remove non ascii char
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    #Limiting all the  repetitions to two characters.
    text = re_repetition.sub(r"\1\1", text)
    #Remove single characters
    text= re.sub(r"\b[a-zA-Z]\b", '', text)
    #Remove newline, return, tab, form [ \n\r\t\f]
    text = re.sub(r'(\s+)', ' ', text)
    #Strip whitespaces at the beginning and at the end of text
    text = text.strip()
    return text


def preprocess_data(df):
    df = df[df['Abstract'].str.strip() != "?"]
    df['text_cleaned'] = df['Abstract'].apply(cleaning)
    df = df.reset_index(drop=True)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    df['text_cleaned'] = df['text_cleaned'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(x) if w.isalpha() and w not in stop_words])
    )
    return df

def model_topics(texts):
    topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", nr_topics="auto")
    start = time.time()
    topics, probs = topic_model.fit_transform(texts)
    end = time.time()
    total_time = round((end - start) / 60)
    print("\ncomp_time in min:\t" + str(total_time))
    return topic_model, total_time

def plot_topics(topic_model, unique_topics):
    nrows = int(np.ceil(len(unique_topics) / 2))
    fig, axs = plt.subplots(nrows, 2, figsize=(15, nrows*5))
    if nrows == 1:
        axs = axs.flatten()
    for idx, topic in enumerate(unique_topics):
        topic_words = topic_model.get_topic(topic)[:10]
        words = [word for word, _ in topic_words]
        scores = [score for _, score in topic_words]
        row_idx = idx // 2
        col_idx = idx % 2
        axs[row_idx, col_idx].barh(words, scores)
        axs[row_idx, col_idx].set_title(f'Topic {topic}')
        axs[row_idx, col_idx].invert_yaxis()
    if len(unique_topics) % 2 != 0:
        fig.delaxes(axs[-1, -1])
    plt.tight_layout()
    plt.show()
    
def prepare_data_for_coherence(topic_model, documents_per_topic):
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    return tokens, dictionary, corpus

def evaluate_coherence(topic_model, topics, tokens, dictionary, corpus):
    topic_words = [[words for words, _ in topic_model.get_topic(topic)]
                   for topic in range(len(set(topics))-1)]
    coherence_model_cv = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()

    coherence_model_uci = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary, coherence='c_uci')
    coherence_uci = coherence_model_uci.get_coherence()
    
    # Se la misura `c_uci` è quella desiderata, non è necessario calcolare `c_mass`
    # coherence_model_mass = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary, coherence='c_mass')
    # coherence_mass = coherence_model_mass.get_coherence()

    return coherence_cv, coherence_uci #, coherence_mass

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    topic_model, total_time = model_topics(df['text_cleaned'])
    topic_info = topic_model.get_topic_info()
    topic_info_pd = pd.DataFrame(topic_info)

    unique_topics = topic_info_pd['Topic'].unique()
    plot_topics(topic_model, unique_topics)
    topic_model.visualize_topics()
    
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    tokens, dictionary, corpus = prepare_data_for_coherence(topic_model, documents_per_topic)
    coherence_cv, coherence_uci = evaluate_coherence(topic_model, topics, tokens, dictionary, corpus)
    
    print(coherence_cv)
    print(coherence_uci)

    topic_model.generate_topic_labels()
    topic_model.get_representative_docs(4)
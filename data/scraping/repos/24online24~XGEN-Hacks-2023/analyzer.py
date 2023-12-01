from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy
from spacy.lang.ro.examples import sentences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


import re


nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('romanian')


DATA_PATH = './data'


def analyze():
    """
    Analizarea pdf-ului si extragerea cuvintelor cheie
    """
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    # eliminarea numerelor si a spatiilor libere de pe fiecare pagina
    for doc in documents:
        content = re.sub(r'\s+', ' ', doc.page_content)
        content = content.replace('\n', '').replace('\r', '')
        content = re.sub(r'^[0-9]+\s', '', content, flags=re.MULTILINE)

        if content.strip():
            doc.page_content = content
        else:
            documents.remove(doc)

    # tokenizarea cuvintelor si eliminarea cuvintelor care nu sunt alfanumerice
    tokens = []
    for doc in documents:
        words = word_tokenize(doc.page_content)
        tokens += [word.lower() for word in words if word.isalpha()]

    # eliminarea cuvintelor stop words (irelevante)
    custom_words = []
    with open('stopwords/geonetwork-rum.txt', 'r') as f:
        custom_words += [line.strip() for line in f.readlines()]

    tokens = [word for word in tokens if word not in stop_words]
    tokens = [
        word for word in tokens if word not in custom_words and len(word) > 2]

    # calcularea frecventei cuvintelor
    word_freq = Counter(tokens)
    common_words = [word[0] for word in word_freq.most_common(10)]
    print(common_words)

    # Extrage entitatilor din text
    # python -m spacy download ro_core_news_sm!!
    nlp = spacy.load("ro_core_news_sm")
    doc = nlp(sentences[0])
    doc_ro = nlp(" ".join(tokens))
    entities_ro = [ent.text for ent in doc_ro.ents]

    # Aflarea posibilelor teme ale documentului
    text = " ".join(tokens)
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform([text])

    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topic_words = [feature_names[i]
                   for i in lda.components_[0].argsort()[-100:][::-1]]

    print("Cuvinte cheie privind tema cărtii:")
    print(topic_words)

    return common_words, entities_ro, topic_words

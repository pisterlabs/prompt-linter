from typing import List

from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.document_preprocessing import get_stopwords


def documents_to_str(documents: list[Document]):
    return ' '.join(document.page_content for document in documents)


def get_keywords(documents: List[Document], uni=4, bi=4) -> str:
    """
    Returns unigrams and bigrams
    :param documents: list of documents to extract keywords from, will be transformed to single string beforehand
    :param uni: number of unigrams to return
    :param bi: number of bigrams to return
    :return: keywords as string (separated by space)
    """

    text = documents_to_str(documents)

    top_bigrams = extract_keywords_from_text(text, n=bi, ngram_range=(2, 2))
    top_unigrams = [unigram for unigram in extract_keywords_from_text(text, n=uni, ngram_range=(1, 1))
                    if not any(unigram in bigram for bigram in top_bigrams)]
    return ' '.join(top_unigrams + top_bigrams)


def extract_keywords_from_text(text, n=5, ngram_range=(1, 1)):
    kw_scores = get_keyword_scores(text, ngram_range=ngram_range)
    return [kw[0] for kw in sorted(kw_scores.items(), key=lambda item: item[1], reverse=True)][:n]


def get_keyword_scores(text, ngram_range=(1, 1)):
    """
    Returns a dict mapping each unique token to its TF-IDF score
    """
    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words=get_stopwords(["german", "english"])
    )
    # Compute TF-IDF scores
    tfidf_matrix = vectorizer.fit_transform([text])
    # Get the feature names (tokens)
    feature_names = vectorizer.get_feature_names_out()
    # Create a dictionary of token to TF-IDF score
    keyword_scores = {}
    for col in tfidf_matrix.nonzero()[1]:
        keyword_scores[feature_names[col]] = tfidf_matrix[0, col]
    return keyword_scores

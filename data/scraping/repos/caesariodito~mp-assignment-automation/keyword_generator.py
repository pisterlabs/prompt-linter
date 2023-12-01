# Import re, nltk, and WordNetLemmatizer
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import string
import openai

nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text: str, lang: str = 'en') -> str:
    """This function will do a preprocessing of the text input

    Args:
        text (str): text input to preprocess (unpreprocessed text)
        lang (str, optional): Choose language to process the text input. 
        Available languages: 'id','en'. Defaults to 'en'.

    Returns:
        str: preprocessed text
    """

    if lang == 'en':
        stopwords = nltk.corpus.stopwords.words('english')
        lemmatizer = WordNetLemmatizer()

        p_text = re.sub('[^a-zA-Z]', ' ', text)
        p_text = p_text.lower()
        p_text = p_text.split()
        p_text = [lemmatizer.lemmatize(word)
                  for word in p_text if not word in set(stopwords)]
        p_text = ' '.join(p_text)
    elif lang == 'id':
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        p_text = text.translate(str.maketrans(
            '', '', string.punctuation)).lower()
        tokens = word_tokenize(p_text)
        list_stopwords = set(nltk.corpus.stopwords.words('indonesian'))
        removed = []
        for t in tokens:
            if t not in list_stopwords:
                removed.append(t)
        p_text = ' '.join(removed)
        p_text = stemmer.stem(p_text)

    return p_text


def get_keywords(text: str, api_key: str = '', lang: str = 'english') -> str:
    """This function will be used to get the keywords from the preprocessed text

    Args:
        text (str): raw text | unpreprocessed text
        api (str, optional): This is the api key secret to send request to openai API. If there are no api provided, the function will use term frequency method to find keywords. Defaults to ''.
        lang (str, optional): Language to do preprocessing of the text input. Defaults to False.

    Returns:
        list: top 5 keywords from the text input
    """

    if api_key == '':
        api = False
    else:
        api = True

    if api:
        openai.api_key = api_key
        prompt = "Find the most useful terms in " + lang + \
            " language from this text. Only print out the terms. These are the texts: \n\n" + text
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.3,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0
        )

        keywords = response['choices'][0]['text']

    else:
        text = preprocess_text(text, lang)
        term_frequencies = Counter(text.split())
        potential_words = term_frequencies.most_common()[:5]

        keywords = []
        for word, _ in potential_words:
            keywords.append(word)
        # print(keywords)

    return str(keywords)

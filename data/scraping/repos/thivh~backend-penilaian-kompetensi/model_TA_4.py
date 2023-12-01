import re
import string
from torch import clamp
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk
import cohere
import time

nltk.download("punkt")
nltk.download("stopwords")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import copy
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


modelName = "cohere"
co = cohere.Client("C9p0PVtj41ZlA1Ay81UjRQ3vvfLcsMlEQph1a2Mv")

apiCount = 0
apiCount2 = 0
factory = StemmerFactory()
stemmer = factory.create_stemmer()
listStopword = set(stopwords.words("indonesian"))

def check_similarity(
    text,
    kamus,
    number1=1,
    punctuation1=1,
    lower1=1,
    stem1=1,
    stopword1=0,
    number2=1,
    punctuation2=1,
    lower2=1,
    stem2=1,
    stopword2=1,
    knn=5,
    word1=[],
):
    global apiCount, apiCount2
    teks = text
    kamus2 = copy.deepcopy(kamus)
    # kalimat = sent_tokenize(
    #     teks,
    #     # language='indonesian'
    # )  # split text into sentences
    for word3 in kamus.keys() if len(word1) == 0 else word1:
        for word4 in kamus[word3].keys():
            sentences = []
            sentences.append(kamus[word3][word4])

            # for k in kalimat:
            #     sentences.append(k)

            sentences.append(teks)

            for i in range(1):
                sentences[i] = (
                    re.sub(r"\d+", "", sentences[i]) if number1 == 1 else sentences[i]
                )  # remove numbers
                sentences[i] = (
                    sentences[i].translate(
                        str.maketrans(string.punctuation, " " * len(string.punctuation))
                    )
                    if punctuation1 == 1
                    else sentences[i]
                )  # remove punctuation
                sentences[i] = (
                    sentences[i].lower() if lower1 == 1 else sentences[i]
                )  # lower case
                sentences[i] = (
                    stemmer.stem(sentences[i]) if stem1 == 1 else sentences[i]
                )  # stemming

                if stopword1 == 1:
                    sentences[i] = word_tokenize(sentences[i])  # tokenization
                    sentences[i] = [
                        word for word in sentences[i] if not word in listStopword
                    ]  # remove stopwords
                    sentences[i] = " ".join(sentences[i])  # join words

                sentences[i] = sentences[
                    i
                ].strip()  # remove leading and trailing whitespace
                sentences[i] = re.sub(r"\s+", " ", sentences[i])  # remove extra space

            for i in range(1, len(sentences)):
                sentences[i] = (
                    re.sub(r"\d+", "", sentences[i]) if number2 == 1 else sentences[i]
                )
                sentences[i] = (
                    sentences[i].translate(
                        str.maketrans(string.punctuation, " " * len(string.punctuation))
                    )
                    if punctuation2 == 1
                    else sentences[i]
                )
                sentences[i] = sentences[i].lower() if lower2 == 1 else sentences[i]
                sentences[i] = (
                    stemmer.stem(sentences[i]) if stem2 == 1 else sentences[i]
                )

                if stopword2 == 1:
                    sentences[i] = word_tokenize(sentences[i])  # tokenization
                    sentences[i] = [
                        word for word in sentences[i] if not word in listStopword
                    ]  # remove stopwords
                    sentences[i] = " ".join(sentences[i])  # join words

                sentences[i] = sentences[
                    i
                ].strip()  # remove leading and trailing whitespace
                sentences[i] = re.sub(r"\s+", " ", sentences[i])  # remove extra space
            # print(sentences)

            # get sentence embeddings
            if(apiCount >= 100):
                apiCount = 0
                time.sleep(60)

            embeddings = co.embed(
                texts=sentences,
                model="embed-multilingual-v2.0",
                truncate="END"
                ).embeddings
            apiCount += 1
            apiCount2 += 1

            # calculate
            x = cosine_similarity([embeddings[0]], embeddings[1:])

            # print("kompetensi: " + word1 + ", level: " + word2)

            x = x[0]
            # print(x)
            x.sort()
            # print(x[-knn:])
            if len(x) < knn or knn < 1:
                result = sum(x) / len(x)
            else:
                result = sum(x[-knn:]) / len(x[-knn:])
            # print(result)
            kamus2[word3][word4] = round(result,3)
    return kamus2

# res = check_similarity(teks, kamus)

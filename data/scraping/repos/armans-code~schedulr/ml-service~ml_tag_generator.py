import os

import openai
import json
import random
# If it doesn't work, open "Install Certificates.command"
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')
from collections import defaultdict
import pprint

openai.api_key = os.environ['OPENAI_API_KEY']

def text_parser_synonym_finder(text: str):
    tokens = word_tokenize(text)
    # print(tokens)
    synonyms = defaultdict(list)

    for token in tokens:
        for syn in wordnet.synsets(token):
            for i in syn.lemmas():
                synonyms[token].append(i.name())

    return synonyms


def generate_tags(textInput):
    # Keywords Parameters
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Extract keywords from this text:\n\n" + textInput,
        temperature=0.3,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0
    )

    # Find all Keywords/Phrases
    keywords = response["choices"][0]["text"].lower()
    index = keywords.index("\n\n")
    keywords = keywords[index + 2:]

    # print(keywords)

    # Move Keywords into Array
    keywordsArray = keywords.split()

    for x in range(len(keywordsArray)):
        keywordsArray[x] = ''.join(letter for letter in keywordsArray[x] if letter.isalnum())

    keywordsNoFormatting = ""

    for x in range(len(keywordsArray)):
        keywordsNoFormatting += keywordsArray[x] + " "

    keywordsNoFormatting = keywordsNoFormatting.strip()

    # print(keywordsArray)

    # Run Function to find synonyms with keywords
    synonymsDictionary = text_parser_synonym_finder(text=keywordsNoFormatting)

    # print(synonymsDictionary)
    # print(keywordsArray)

    # print(keywordsArray)

    # x is keyword; add 2 synonyms from each word
    shortlistSynonyms = []

    for x in keywordsArray:
        # Remove Duplicate Synonyms
        synonymsOfWord = synonymsDictionary.get(x)

        allSynonyms = {}

        if synonymsOfWord is not None:
            allSynonyms = list(dict.fromkeys(synonymsDictionary.get(x)))

        for y in range(len(allSynonyms)):
            allSynonyms[y] = allSynonyms[y].lower()

        #print(allSynonyms)

        # Remove Key Word is Present
        if x in allSynonyms:
            allSynonyms.remove(x)

        # Get 2 random synonyms if 2 or more synonyms present, get 1 synonym if 1 present, leave alone if 0 synonyms
        if len(allSynonyms) >= 2:
            firstRandom = random.randint(0, len(allSynonyms) - 1)
            secondRandom = firstRandom
            while secondRandom == firstRandom:
                secondRandom = random.randint(0, len(allSynonyms) - 1)
            shortlistSynonyms.append(allSynonyms[firstRandom])
            shortlistSynonyms.append(allSynonyms[secondRandom])

        elif len(allSynonyms) >= 1:
            shortlistSynonyms.append(allSynonyms[0])

        # print(allSynonyms)

    # print(shortlistSynonyms)

    allKeywordsAndRelated = []

    for x in keywordsArray:
        allKeywordsAndRelated.append(x)

    for x in shortlistSynonyms:
        allKeywordsAndRelated.append(x)

    return allKeywordsAndRelated
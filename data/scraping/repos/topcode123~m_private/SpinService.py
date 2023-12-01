import pickle
import random
from underthesea import word_tokenize as word_tokenize_vi

from nltk.corpus import wordnet
from random import randint
import nltk.data
from bs4 import BeautifulSoup as soup
from nltk.tokenize import word_tokenize as word_tokenize_en
import re

import openai

from pymongo import MongoClient
from Settings import CONNECTION_STRING_MGA1

users = MongoClient(CONNECTION_STRING_MGA1).accounts.data


class SpinService:

    def __init__(self) -> None:
        with open("dataspin.p", "rb") as file:
            self.dataspin = pickle.load(file)
        nltk.download('omw-1.4')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        wordnet.ensure_loaded()
        self.type_soup = "html.parser"

    @staticmethod
    def rewrite_article_gpt3(raw_data, lang):
        api_key = users.find_one({"username": 'KenLil'})
        if not api_key:
            raise ValueError("Missing OPEN AI API KEY")
        openai.api_key = api_key.get("apiKey")

        promt = "in a suspensful and mysterious style rewrite text below\n" + raw_data + ""
        if lang == "vi":
            promt = "viết lại đoạn văn sau bằng tiếng việt\n" + raw_data + ""

        results = openai.Completion.create(
            model="text-davinci-003",
            prompt=promt,
            temperature=0,
            max_tokens=len(raw_data) + 1,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0
        )
        response = dict(results)
        openai_response = response['choices']
        return openai_response[-1]['text']

    def spin_paragraph(self, p_paragraph1, keyword, userId):
        p_paragraph = [str(t) for t in p_paragraph1.contents]
        word_splits = []
        print("keyword: ", keyword)
        if userId == "615d1d3f570562748141c73e -------------":
            print("use open ai to process")
            paragraph = soup(self.rewrite_article_gpt3(str(p_paragraph1), "vi"), self.type_soup)
            return paragraph

        try:
            for i in p_paragraph:
                if not re.match(r'<[^>]+>', i):
                    word_splits = word_splits + word_tokenize_vi(i)
                else:
                    word_splits = word_splits.append(i)
                    if re.match(r'<img [^>]+>', i):
                        word_splits = word_splits.append("<br>")

            if word_splits is not None:
                for index_word in range(len(word_splits)):
                    if word_splits[index_word] in self.dataspin and word_splits[
                        index_word].lower() not in keyword.lower():
                        word_splits[index_word] = random.choice(self.dataspin[word_splits[index_word]])
                paragraph = " ".join(word_splits)
                paragraph = soup(paragraph, self.type_soup)

                return paragraph
            else:
                return p_paragraph1
        except:
            return p_paragraph1

    def spin_paragraph_en(self, p_paragraph1, keyword, userId):

        p_paragraph = [str(t) if not re.match(r'<[^>]+>', str(t)) else str(t) for t in p_paragraph1.contents]
        if userId == "615d1d3f570562748141c73e ---------------":
            print("use open ai to process")
            paragraph = soup(self.rewrite_article_gpt3(str(p_paragraph1), "en"), self.type_soup)
            return paragraph

        output = ""

        # Get the list of words from the entire text
        # try:
        words = []
        for i in p_paragraph:
            if i != None:
                if not re.match(r'<[^>]+>', i) and i.lower() not in keyword.lower():
                    try:
                        words = words + word_tokenize_en(i)
                    except:
                        pass
                else:
                    words = words + [i]
        # except:
        # except Exception as e:
        #     print(e)
        #     return p_paragraph1

        if words != None:
            if len(words) > 0:
                tagged = nltk.pos_tag(words)
            for i in range(0, len(words)):
                replacements = []
                # if (tagged[i][1] == 'NN' or tagged[i][1] == 'JJ' or tagged[i][1] == 'RB') and not re.match(r'<[^>]+>',
                #                                                                                            words[i]):
                try:
                    for syn in wordnet.synsets(words[i]):

                        word_type = tagged[i][1][0].lower()
                        if syn.name().find("." + word_type + "."):
                            # extract the word only
                            r = syn.name()[0:syn.name().find(".")]
                            replacements.append(r)

                except Exception as e:
                    pass

                if len(replacements) > 0:
                    # Choose a random replacement
                    replacement = replacements[randint(0, len(replacements) - 1)]
                    output = output + " " + replacement.replace("_", " ")
                else:
                    # If no replacement could be found, then just use the
                    # original word
                    output = output + " " + words[i]
        else:
            return p_paragraph1
        output = soup(output, self.type_soup)
        return output

    def spin_title_vi(self, p_paragraph1, keyword):
        aaa = word_tokenize_vi(p_paragraph1)

        try:
            word_splits = aaa
            if (word_splits != None):
                for index_word in range(len(word_splits)):
                    if word_splits[index_word] in self.dataspin and word_splits[index_word].lower() not in keyword:
                        word_splits[index_word] = random.choice(self.dataspin[word_splits[index_word]])
                paragraph = " ".join(word_splits)
                paragraph
                return paragraph
            else:
                return p_paragraph1
        except:
            return p_paragraph1

    def spin_title_en(self, p_paragraph1, keyword):
        aaa = word_tokenize_en(p_paragraph1)

        output = ""

        # Get the list of words from the entire text
        try:
            words = aaa
        except:
            return p_paragraph1

        if words != None:
            if len(words) > 0:
                tagged = nltk.pos_tag(words)
            for i in range(0, len(words)):
                replacements = []
                if (tagged[i][1] == 'NN' or tagged[i][1] == 'JJ' or tagged[i][1] == 'RB') and not re.match(r'<[^>]+>',
                                                                                                           words[i] and
                                                                                                           words[
                                                                                                               i].lower() not in keyword.lower()):
                    try:
                        for syn in wordnet.synsets(words[i]):

                            word_type = tagged[i][1][0].lower()
                            if syn.name().find("." + word_type + "."):
                                # extract the word only
                                r = syn.name()[0:syn.name().find(".")]
                                replacements.append(r)

                    except Exception as e:
                        pass

                if len(replacements) > 0:
                    # Choose a random replacement
                    replacement = replacements[randint(0, len(replacements) - 1)]
                    output = output + " " + replacement.replace("_", " ")
                else:
                    # If no replacement could be found, then just use the
                    # original word
                    output = output + " " + words[i]
        else:
            return p_paragraph1
        return output

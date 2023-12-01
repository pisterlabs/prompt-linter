from enum import Enum
from typing import List
import openai
import os
import json
import string
import re

from gpt_translate.articles.dto import LanguageEnum
from pydantic import BaseModel


openai.api_key = os.environ["OPENAI_API_KEY"]

class ArticleManager:
    def __init__(self):
        pass

    def search_by_tags(self, input_tag: str, language: LanguageEnum):
        raise NotImplementedError()

    def search_by_text(self, input_str: str, language: LanguageEnum):
        raise NotImplementedError()

    def search_by_embedding(self, input_str: str, language: LanguageEnum):
        raise NotImplementedError()

    def get_embedding(self, text: str, model="text-embedding-ada-002"):
        try:
            text = text.replace("\n", " ")
            return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
        except Exception as e:
            print(e)
            return []
    
    def add_complete_article(self):
        """
        add article to current existing articles, persist

        article should be a dictionary or pydantic model with following fields minimum:
        title
        english_title
        english_tags
        chinese_tags
        text
        translation
        embedding (15XX dim dimensions)

        article should have column ID which should be unique
        """
        raise NotImplementedError()
    
    def complete_article(self):
        """
        [modify DB]
        given an article with only
        title
        text,

        return a dictionary s.t
        english_title
        translation
        english_title
        english_tags
        chinese_tags
        """
        raise NotImplementedError()
    
    def add_tag(self, article_id: int):
        """
        [modify DB]
        add english or chinese tags to an article
        """
        raise NotImplementedError()


    def preprocess_tags(self, tags: List[str], delimiter: str=" "):
        # # Concatenate the tags into a single string
        # tag_string = ' '.join(tags).lower()

        # # Preprocess the tag string by replacing spaces in multi-word tags with underscores
        # tag_string = re.sub('[%s]' % re.escape(string.punctuation), '', tag_string)
        # tag_string = re.sub(r'(\b\w+\s+\w+\b)', lambda m: m.group().replace(" ", "_"), tag_string)

        # return tag_string
        """
        Concatenates a list of words with the specified delimiter.
        If a word contains spaces, replaces them with underscores before concatenating.
        """
        # Replace spaces with underscores in any multi-word strings
        words = [w.replace(' ', '_') if ' ' in w else w for w in tags]
        
        # Concatenate the words with the specified delimiter
        return delimiter.join(words)
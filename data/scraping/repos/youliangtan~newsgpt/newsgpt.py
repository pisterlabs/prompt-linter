#!/usr/bin/env python3

import os
import openai
from langchain.llms import AzureOpenAI
from typing import List
import time

from langchain.document_loaders import SeleniumURLLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate

##############################################################################


class NewsCategory:
    ALL = "all"
    BUSINESS = "business"
    POLITICS = "politics"
    SPORTS = "sports"
    TECHNOLOGY = "technology"


class NewsLength:
    SHORT = 10
    MEDIUM = 20
    LONG = 30


##############################################################################

class NewsGPT:
    def __init__(self,
                 api_key: str,
                 api_type: str,
                 api_base: str,
                 api_version: str,
                 model_name: str,
                 max_tokens: int = 1000
                 ):
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_TYPE"] = api_type
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_API_VERSION"] = api_version
        openai.api_key = api_key
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_type = api_type

        self.llm = AzureOpenAI(
            deployment_name=model_name,
            model_name=model_name,
            max_tokens=max_tokens)  # default is 16 in openai API
        print("Done Initializing NewsGPT")

    def get_urls(
            self, category: NewsCategory
        ) -> List[str]:
        """
        Get the news site urls for the given category
        """
        _urls = ["https://www.wsj.com/", "https://www.nytimes.com/", "https://www.apnews.com/", "https://www.bbc.com/"]
        post_fixes = {
            NewsCategory.ALL: ["", "", "", "news/"],
            NewsCategory.BUSINESS: ["news/business", "section/business", "hub/business", "news/business"],
            NewsCategory.TECHNOLOGY: ["news/technology", "section/technology", "hub/technology", "news/technology"],
            NewsCategory.POLITICS: ["news/politics", "section/politics", "hub/politics", ""],
            NewsCategory.SPORTS: ["news/sports", "section/sports", "hub/sports", "sport"],
        }
        # append urls
        for i in range(len(_urls)):
            _urls[i] += post_fixes[category][i]
        return _urls

    def get_news(
        self, category: NewsCategory, chars_limit=3000
    ) -> List[Document]:
        """
        Returns a list of documents from the given category
        """
        start = time.time()
        urls = self.get_urls(category)
        print(urls)
        loader = SeleniumURLLoader(urls=urls)
        docs = loader.load()

        # Docs Filtering
        print(len(docs))
        print(type(docs[0]))

        # This tries to truncate the page content to char_limit characters
        for d in docs:
            l = len(d.page_content)
            print(l)
            if l > chars_limit:
                d.page_content = d.page_content[:chars_limit]

        print(" size after truncation of docs")
        for d in docs:
            print(len(d.page_content))
        
        print(f"Time taken to load {len(docs)} docs: {time.time() - start}")
        return docs

    def summarize_docs(
            self,
            docs: List[Document],
            news_category: NewsCategory,
            news_length: NewsLength,
            single_doc: bool = False,
        ):
        """
        Summarize the given list of documents
        if single_doc is False: use map-reduce summarization
        else:                   use single doc summarization
        """
        print("Summarizing docs")
        MAP_REDUCE_DEBUG = False
        start = time.time()
        
        if news_category == NewsCategory.ALL:
            news_category = ""

        map_prompt_template = f"Write a {news_category} news headlines summary of the following:"
        map_prompt_template += " \n {text} \n"
        map_prompt_template += f"PROVIDE SUMMARY WITH AROUND {news_length + 10} SENTENCES"

        reduce_prompt_template = f"Write summary with {news_length} bullet points for today's"
        reduce_prompt_template += f" {news_category} headlines from multiple News Sources:"
        reduce_prompt_template += " \n {text} \n"
        reduce_prompt_template += f"PROVIDE SUMMARY WITH {news_length} BULLET POINTS"

        MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        REDUCE_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        if single_doc:
            chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=REDUCE_PROMPT)
            summary = chain.run(docs)

        # use map-reduce summarization
        elif MAP_REDUCE_DEBUG:
            chain = load_summarize_chain(self.llm,
                                         chain_type="map_reduce",
                                         map_prompt=MAP_PROMPT,
                                         combine_prompt=REDUCE_PROMPT,
                                         return_map_steps=True)
            summary = chain({"input_documents": docs}, return_only_outputs=True)
        else:
            chain = load_summarize_chain(
                self.llm, chain_type="map_reduce", map_prompt=MAP_PROMPT,
                combine_prompt=REDUCE_PROMPT)
            summary = chain.run(docs)

        print(f"Time taken to summarize {len(docs)} docs: {time.time() - start}")        
        return summary

    def summarize(
                self,
                news_category: NewsCategory,
                news_length: NewsLength
            ) -> str:
        docs = self.get_news(category=news_category)
        return self.summarize_docs(docs, news_category, news_length)


##############################################################################

if __name__ == "__main__":
    _api_key = "700fa82411ad46069807d49abd48c7ad"
    _api_base =  "https://newsgpt.openai.azure.com/"
    _api_type = 'azure'
    _api_version = '2022-12-01'
    _model_name = "text-davinci-003"
    
    newsgpt = NewsGPT(api_key=_api_key,
                        api_type=_api_type,
                        api_base=_api_base,
                        api_version=_api_version,
                        model_name=_model_name)
    result = newsgpt.summarize(
        news_category=NewsCategory.BUSINESS,
        news_length=NewsLength.MEDIUM)
    
    print("\n\n ---------------- That's the result: \n")
    print(result)

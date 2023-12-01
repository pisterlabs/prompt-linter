#!/usr/bin/env python3

from lib.newsgpt import NewsGPT, NewsCategory, NewsLength
from lib.summarydb import SummaryDB
from lib.news_scrapper import get_content, get_news

from langchain.docstore.document import Document

##############################################################################


def populate_db(use_seleluim=False):
    _api_key = "700fa82411ad46069807d49abd48c7ad"
    _api_base = "https://newsgpt.openai.azure.com/"
    _api_type = 'azure'
    _api_version = '2022-12-01'
    _model_name = "text-davinci-003"

    if use_seleluim:
        max_output_tokens = 1000
    else:
        max_output_tokens = 1500

    newsgpt = NewsGPT(api_key=_api_key,
                      api_type=_api_type,
                      api_base=_api_base,
                      api_version=_api_version,
                      model_name=_model_name,
                      max_tokens=max_output_tokens,)

    for cat in [
        NewsCategory.ALL, NewsCategory.BUSINESS, NewsCategory.POLITICS,
        NewsCategory.SPORTS, NewsCategory.TECHNOLOGY
    ]:
        print("Generating summary for category: ", cat)

        if use_seleluim:
            docs = newsgpt.get_news(category=cat)
        else:
            urls = get_news(NewsCategory.ALL)
            c = get_content(urls, cat)
            docs = [Document(page_content=c)]

        for news_len in [
            NewsLength.SHORT,
            NewsLength.MEDIUM,
            NewsLength.LONG
        ]:
            res = newsgpt.summarize_docs(docs, cat, news_len,
                                         single_doc=(not use_seleluim))
            print(" Summary:", res, "\n\n")
            db = SummaryDB()
            db.write_summary(cat, news_len, summary=res)
            print("Done writing to DB.")

##############################################################################


if __name__ == "__main__":
    populate_db()

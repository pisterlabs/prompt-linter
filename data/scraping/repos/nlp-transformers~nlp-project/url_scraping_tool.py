from langchain.tools import Tool
from langchain.document_loaders import SeleniumURLLoader
from summarize_docs import summarize_docs
def url_scraper(url) :
    # print(query)
    urls = [
        #url,
        "https://news.google.com",
        "https://www.marketwatch.com",
    ]
    loader = SeleniumURLLoader(urls=urls)
    docs = loader.load()
    #print(docs)
    summary_text = summarize_docs(docs)

    result = {
        "tool": "url_scraping",
        "tldr": summary_text,
        "article": docs
    }
    return result

url_scraping_tool = Tool.from_function(
    func = url_scraper,
    name = "url scraping",
    return_direct=True,
    description="Use this tool to obtain the latest headlines or news by scraping content from the URL https://news.google.com. Prefer this over normal search when searching for news.",
)
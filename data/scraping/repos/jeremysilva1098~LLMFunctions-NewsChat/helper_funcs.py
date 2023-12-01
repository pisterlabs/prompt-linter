from langchain.document_loaders import UnstructuredURLLoader
import json



def get_url_content(url: str) -> str:
    loader = UnstructuredURLLoader(urls=[url])
    data = loader.load()
    return data[0].page_content


def add_url_content_to_news_res(news_articles: list) -> list:
    articlesCleaned = []
    hits = 0
    while hits < 5:
        for article in news_articles:
            try:
                urlContent = get_url_content(article['url'])
                article['content'] = urlContent
                articlesCleaned.append(article)
                hits += 1
            except:
                print("Failed to get url content for: ", article['url'])
                pass
        break
    #print("Clean Articles: ", json.dumps(articlesCleaned, indent=4))
    return articlesCleaned


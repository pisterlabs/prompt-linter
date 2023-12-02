import trafilatura
import json
import openai
import simple_colors  
from time import sleep
import re



def accurate(topic, searchResults):
    articles = getArticles(searchResults)
    print(simple_colors.yellow(f"total articles: {len(articles)}\n"))
    bestArticles = curator(topic, articles)
    return findArticles(articles, bestArticles)
    


def findArticles(articles, searchArticles):
    findedArticles = []
    for search in searchArticles:
        findedArticles.append(list(filter(lambda articles: articles['link'] == search, articles))[0])

    return findedArticles



def getArticles(searchResults):
    print(simple_colors.yellow("\nExtracting content from urls (this may take a few minutes)..."))

    content = []

    for searchResult in searchResults:
        for result in searchResult["searchResults"]:
            extract = extractContent(result)
            if extract["title"] != "" and extract["pagetype"] == "article":
                content.append(extract)

    # for searchResult in searchResults:
    #     for result in searchResult["SearchNewsResults"]:
    #         extract = extractContent(result)
    #         if extract["title"] != "" and extract["pagetype"] == "article":
    #             content.append(extract)

    return content



def extractContent(url):
    downloaded = trafilatura.fetch_url(url)

    data = trafilatura.extract(downloaded, output_format="json", include_comments=False)

    try:
        content = json.loads(data)
        return {"link":url, "title":content["title"], "author":content["author"], "content":content["raw_text"], "pagetype":content["pagetype"]}
    except Exception:
        return {"link":"", "title":"", "author":"", "content":""}
    


def curator(topic, articles):
    print(simple_colors.yellow("Finding the best articles (this may take a few minutes)..."))

    prompt = []
    bestArticlesLinks = []
    bestArticles = []

    tokens = 0
    c = 0

    while len(articles) > 10 or len(articles) < 1:
        for article in articles:

            c += 1

            formatedArticle = articleFormat(article)

            tokens += calculateTokens(f"Artigo {c}\n{formatedArticle}")

            if tokens > 2500:
                prompt.insert(0, {"role": "system", "content": f'You are a bot that specializes in curating articles about {topic} sent to you by another bot called a search bot. You read the content of the articles found by him and do a thorough analysis of the text. At the end of your evaluation you always return only an unnumbered list separated by "* " with the link of the most relevant article provided and that have more relation to {topic} and no comments'})
                
                try:
                    result = accurateBot(prompt)
                    if result != []:
                        bestArticles.append(result)
                except Exception:
                    pass
                
                prompt = []
                tokens = 0
                prompt.append({"role":"system", "content":f"Article {c}\n"+formatedArticle})

            else:
                prompt.append({"role":"system", "content":f"Article {c}\n"+formatedArticle})

        bestArticles = cleanResults(articles, bestArticles)
        articles = findArticles(articles, bestArticles)
        print(simple_colors.yellow(f"total articles: {len(bestArticles)}"))
    else:
        bestArticles = articles
        
    print(simple_colors.yellow(f"total articles: {len(bestArticles)}"))
    print(simple_colors.yellow(f"\nbest articles: {bestArticles}"))

    return bestArticles



def accurateBot(prompt):
    prompt.append({"role": "user", "content": f'list the best links'})

    acuratedArticles = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = acuratedArticles.choices[0].message["content"]

    sleep(2)

    return parseArticles(response)
    


def cleanResults(articles, data):
    url_extract_pattern = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
    newData = []

    for i in data:
        for j in i:
            try:
                pureLink = re.findall(url_extract_pattern,j)[0]
            except Exception:
                pureLink = re.findall(url_extract_pattern,j)

            for article in articles:
                if pureLink in article.values():
                    newData.append(pureLink)
                    break

    return newData



def calculateTokens(prompt_tokens):
    return len(prompt_tokens)/4



def articleFormat(articleData):
    article = ''
    article += f'link: {articleData["link"]}\n'
    article += f'{articleData["title"]}\n{articleData["author"]}\n{articleData["content"]}'
    return article



def parseArticles(articles):
    text = articles.replace("\n", "")
    text = text.split("*")[1:]
    return text
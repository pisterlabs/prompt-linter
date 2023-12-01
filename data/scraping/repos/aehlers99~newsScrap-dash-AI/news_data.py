import requests
from bs4 import BeautifulSoup
import openai

# Brazil Journal
def brazilJournal_main(headers):
    # Main headlines
    headlines_list = []
    headlines_href_list = []

    URL = "https://braziljournal.com/"

    r = requests.get(URL, headers=headers)

    soup = BeautifulSoup(r.content, "html.parser")

    headlines = soup.find_all("h2", class_="boxarticle-infos-title")

    # Procurando por elementos filhos nas manchetes
    for headline in headlines:
        headlines_href = headline.find("a")['href']
        headlines_text = headline.find("a").text
        # Formatando o texto
        headlines_text = headlines_text.strip()
        headlines_text = headlines_text.replace("  ", "")
        headlines_text = headlines_text.replace("BREAKING:", "")
        headlines_text = headlines_text.replace("EXCLUSIVO:", "")
        headlines_list.append(headlines_text)
        headlines_href_list.append(headlines_href)

    dic_test = dict(zip(headlines_list, headlines_href_list))

    return dic_test

def brazilJournal_economy(headers):
    URL = "https://braziljournal.com/category/economia/"

    r = requests.get(URL, headers=headers)

    soup = BeautifulSoup(r.content, "html.parser")

    headlinesEconomy = soup.find_all("h2", class_="boxarticle-infos-title")

    headlines_list = []
    headlines_href_list = []

    for headline in headlinesEconomy:
        headlinesEconomy_href = headline.find("a")['href']
        headlinesEconomy_text = headline.find("a").text
        # Formating text and printing
        headlinesEconomy_text = headlinesEconomy_text.strip()
        headlinesEconomy_text = headlinesEconomy_text.replace("BREAKING:", "")
        headlinesEconomy_text = headlinesEconomy_text.replace("EXCLUSIVO:", "")
        headlinesEconomy_text = headlinesEconomy_text.replace("  ", "")
        headlines_list.append(headlinesEconomy_text)
        headlines_href_list.append(headlinesEconomy_href)

    dic_test = dict(zip(headlines_list, headlines_href_list))

    return dic_test

def brazilJournal_business(headers):
    # headlinesBusiness for business
    URL = "https://braziljournal.com/categoria/negocios/"

    r = requests.get(URL, headers=headers)

    soup = BeautifulSoup(r.content, "html.parser")

    headlinesBusiness = soup.find_all("h2", class_="boxarticle-infos-title")

    headlines_list = []
    headlines_href_list = []

    for headline in headlinesBusiness:
        headlinesBusiness_href = headline.find("a")['href']
        headlinesBusiness_text = headline.find("a").text
        # Formating text and printing
        headlinesBusiness_text = headlinesBusiness_text.strip()
        headlinesBusiness_text = headlinesBusiness_text.replace("BREAKING:", "")
        headlinesBusiness_text = headlinesBusiness_text.replace("EXCLUSIVO:", "")
        headlinesBusiness_text = headlinesBusiness_text.replace("  ", "")
        headlines_list.append(headlinesBusiness_text)
        headlines_href_list.append(headlinesBusiness_href)

    dic_test = dict(zip(headlines_list, headlines_href_list))

    return dic_test

# VALOR

def get_headlines(url, headers, class_name=None, class_filter=None):
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, "html.parser")

    if class_name:
        headlines = soup.find_all("h2", class_=class_name)
    elif class_filter:
        headlines = soup.find_all("a", class_=class_filter)
    else:
        return {}

    headlines_list = []
    headlines_href_list = []

    for headline in headlines:
        headlines_href = headline['href']
        headlines_text = headline.text
        # Formatting the text
        headlines_text = headlines_text.strip().replace("  ", "").replace("BREAKING:", "").replace("EXCLUSIVO:", "")
        headlines_list.append(headlines_text)
        headlines_href_list.append(headlines_href)

    if class_filter:
        headlines_list_updated = []
        hyperlink_list_updated = []
        for i in range(len(headlines_list)):
            if not headlines_list[i].strip().startswith("— Em"):
                headlines_list_updated.append(headlines_list[i])
                hyperlink_list_updated.append(headlines_href_list[i])
        headlines_list = headlines_list_updated
        headlines_href_list = hyperlink_list_updated

    return dict(zip(headlines_list, headlines_href_list))

def valorFinances_main(headers):
    return get_headlines("https://valor.globo.com/financas/", headers, class_filter='feed-post-link')

def valorEmpresas_main(headers):
    return get_headlines("https://valor.globo.com/empresas/", headers, class_filter='feed-post-link')

def valorPublic_main(headers):
    return get_headlines("https://valor.globo.com/", headers, class_filter='bstn-dedupe-url')

def valorPaid_main(headers):
    return get_headlines("https://valor.globo.com/", headers, class_filter='bstn-dedupe-url is-subscriber-only')

#Contents

def brazilJournal_news(href, headers):
    URL = href
    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    try:
        headline = soup.find('h1', class_='post-header-title').text
        headline = headline.strip()
    except:
        headline = 'Não foi possível encontrar a manchete'
    try:
        datetime = soup.find('time', class_='post-time').text
        datetime = datetime.strip()
    except:
        datetime = 'Não foi possível encontrar a data e hora'
    try:
        author = soup.find('span', class_='pp-author-boxes-name multiple-authors-name').text
        author = author.strip()
    except:
        author = 'Não foi possível encontrar o autor'
    try:
        article = soup.find('div', class_='post-content-text').text
        article = article.strip()
        article = article.replace('\n', '')
        article = article.replace('\t', '')
        article = article.replace('\r', '')
        article = article.replace('  ', '')
    except:
        article = 'Não foi possível encontrar o artigo'

    return headline, datetime, author, article

def valor_news(href, headers):
    URL = href
    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    try:
        headline = soup.find('h1', class_='content-head__title').text
        headline = headline.strip()
    except:
        headline = 'Não foi possível encontrar a manchete'
    try:
        datetime = soup.find('p', class_="content-publication-data__updated").text
        datetime = datetime.strip()
    except:
        datetime = 'Não foi possível encontrar a data e hora'
    try:
        author = soup.find('p', class_='content-publication-data__from').text
        author = author.strip()
    except:
        author = 'Não foi possível encontrar o autor'
    try:
        article_div = soup.find('div', class_='mc-article-body')
        p_elements = article_div.find_all('p')
        article = ' '.join([p.get_text(strip=True) for p in p_elements])
    except AttributeError:
        article = None

    return headline, datetime, author, article
    
#Open Ai Bulletpoints

def openai_bulletpoints(text : str , headers):

    openai.api_key = "sk-..."

    if openai.api_key is None or openai.api_key == "COLOQUE_SUA_API_KEY_AQUI":
        return "Falha ao identinficar sua openai.api_key. Verifique se você configurou corretamente."

    question = 'Levando em consideração o noticia aseguir, quais são os pontos mais importantes na sua opnião? Resuma exaltando fatores que julgar importantes para economia e em especifico o mercado de crédito. Utilize até 300 tokens em sua resposta e faça tudo no formato de até 4 bullet points. Utilize a seguinte formatação em sua resposta: "-TEXTO;" : ' 

    response = openai.Completion.create(
        engine = 'text-davinci-003',
        prompt = question + text,
        max_tokens = 400,
        top_p = 0.8, 
        temperature = 0.5,
    )

    return response['choices'][0]['text']

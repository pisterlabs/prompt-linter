from django.shortcuts import render
from .forms import MyForm
import cohere
# from openai import OpenAI
from multiprocessing import Process

from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless=new")
driver = webdriver.Chrome(options=chrome_options)

co = cohere.Client('')

returnDict = {}

SENTENCE = 'quels sont les aliments populaires'

blocklist = [
    'style',
    'script',
    'footer',
    # other elements,
]


def parse_content(links: list) -> list:
    data = []
    counter = 0
    for link in links:
        r = requests.get(link)
        soup = BeautifulSoup(r.text, 'html.parser')
        title = soup.find('title').text
        text_elements = [t for t in soup.find_all(text=True) if
                         (t.parent.name not in blocklist and len(t) > 10 and '\n' not in t)]
        text_elements = " ".join(text_elements)

        d = {}
        d["title"] = title
        d["snippet"] = text_elements
        data.append(d)
        if counter == 3: break
        counter += 1

    return data


def write_data_to_txt(d: dict) -> None:
    """
    generate a bunch of txt files in a folder containing all the text for each dictionary
    """
    pass

def search_content(input_str: str) -> list:
    # Query to obtain links
    query = input_str
    links = []  # Initiate empty list to capture final results
    n_pages = 2
    for page in range(1, n_pages):
        url = "http://www.google.com/search?q=" + query + "&start=" + str((page - 1) * 10)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # soup = BeautifulSoup(r.text, 'html.parser')

        search = soup.find_all('div', class_="yuRUbf")
        for h in search:
            links.append(h.a.get('href'))

    return links

def getDocs(string):
    links = search_content(string)
    data = parse_content(links)

    response = co.chat(
        string,
        model="command",
        temperature=0.9,
        documents=data,
        prompt_truncation='AUTO'
    )
    matched_titles = links
    citations = dict(zip(matched_titles, links))
    returnDict["rags"] = response.text
    returnDict['citationsRag'] = citations

def classifyLangauge(string):
    response = co.detect_language(texts=[string])
    language_names = [lang.language_name for lang in response.results]
    returnDict["language"] = language_names[0]
    
def getCoralResponse(string, connector):
    if connector == True:
        response = co.chat(
            string,
            model="command",
            temperature=0.9,
            connectors=[{"id": "web-search"}]

        )

        docs_used = [citation['document_ids'] for citation in response.citations]
        docs_used = [item for sublist in docs_used for item in sublist]
        matched_urls = [doc['url'] for doc in response.documents if doc['id'] in docs_used]
        matched_titles = [doc['title'] for doc in response.documents if doc['id'] in docs_used]
        citations = dict(zip(matched_titles, matched_urls))

        returnDict['citationsNonRag'] = citations
        returnDict["connect"] = response.text
    else:
        response = co.chat(
            string,
            model="command",
            temperature=0.9
        )
        returnDict["noConnect"] = response.text


def my_form_view(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            # Process the form data
            message = form.cleaned_data['message']

            p1 = Process(target=classifyLangauge(message))
            p2 = Process(target=getCoralResponse(message, False))
            p3 = Process(target=getCoralResponse(message, True))
            #p4 = Process(target=getChatGPTResponse(message))
            p6 = Process(target=getDocs(message))

            p1.start()
            p2.start()
            p3.start()
            #p4.start()
            p6.start()

            p1.join()
            p2.join()
            p3.join()
            #p4.join()
            p6.join()
            # Do something with the data (e.g., save to a database)

            # For demonstration purposes, you can just display the data in the template
            return render(request, 'thank_you.html',
                          {'query': message,
                                  'langauge': returnDict["language"],
                                  'noConnect': returnDict["noConnect"],
                                  'withConnect': returnDict["connect"],
                                  'rags': returnDict['rags'],
                                  'citationsNonRag': returnDict["citationsNonRag"],
                                  'citationsRag': returnDict["citationsRag"]})
    else:
        form = MyForm()

    return render(request, 'my_form_view.html', {'form': form})
# TODO: Take average of different values. Iterate until there is confluence.
# TODO: Define a function to convert question into URL. E.g. CISSP certification url -> https://www.isc2.org/Certifications/CISSP
# TODO: Scraping function does not save all text
# TODO: Import as notebook

# for google query search
from googlesearch import search

# scrape_html_only()
from urllib.request import urlopen
from bs4 import BeautifulSoup

# load txt
from langchain.document_loaders import TextLoader

# QA for input document
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# To determine top urls for google query
def find_urls(g_query):
    # prints top 10 search results
    urls = []
    for j in search(g_query, tld="co.in", num=10, stop=10, pause=2):
        print(j)
        urls.append(j)
    # returns list of top 10 search results
    return urls

# Scrape only HTML from a webpage
def scrape_html_only(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    # Save HTML that is scraped
    # Open the file in write mode
    file_name = "scraped.txt"
    file = open(file_name, 'w', encoding='utf-8')

    # Write the string content to the file
    file.write(text)

    # Close the file
    file.close()

    print(f"File '{file_name}' has been saved successfully.")
    return text

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def langchain_response(query):
    # load .txt in the appropriate format for langchain
    loader = TextLoader("scraped.txt")
    documents = loader.load()

    # HTML scraped
    print(documents)

    # split context in order to bypass token limit
    char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200) 
    docs = char_text_splitter.split_documents(documents)
    
    # QA
    # 4 different chain types: "stuff", "map_reduce", "refine", "map_rerank"
    model = load_qa_chain(llm=OpenAI(), chain_type="refine")
    print(model.run(input_documents=docs, question=query))

    # Summary. query is not used in summary
    # model = load_summarize_chain(llm=OpenAI(), chain_type="refine")
    # print(model.run(docs))
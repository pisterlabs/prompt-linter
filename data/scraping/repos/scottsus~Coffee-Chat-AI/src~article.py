from bs4 import BeautifulSoup
import requests
import logging
import re
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.llms import OpenAI
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
# log_file = 'logs/article.log'
# logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

def scrape(url):
    # Send a GET request to the URL and retrieve the HTML content
    response = requests.get(url)
    html_content = response.content

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    for ad in soup.find_all(class_='advertisement'):
        ad.extract()

    # Extract the main content area by selecting specific HTML tags or classes
    main_content = soup.find(class_='article-content')

    # Clean up the extracted content
    clean_text = main_content.get_text(separator='\n')

    # Remove extra whitespace and newlines
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def summarise(scraped):
    text_splitter = CharacterTextSplitter()
    chunks = text_splitter.split_text(scraped)
    summary_chain = load_summarize_chain(OpenAI(temperature=0),
                                            chain_type="map_reduce",verbose=True)  
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
    answer = summarize_document_chain.run(chunks)
    return answer

def questions(scraped):
    text_splitter = CharacterTextSplitter()
    chunks = text_splitter.split_text(scraped)

    #create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    user_question = 'You are a talk show host and youre about to interview a very famous startup founder. Based on the article of this person, generate five potential interesting questions that a wide range of people might find interesting.'
    docs = knowledge_base.similarity_search(user_question)
        
    llm = OpenAI(cache=False)
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
    return response

def article(article_url):
    try:
        scraped = scrape(article_url)
        question = questions(scraped)
        questionlist = re.findall(r'\d+\.\s+(.*)', question)
        
        return questionlist
    
    except Exception:
        print('Article error')
        return []

test_url = 'https://techcrunch.com/2021/09/05/singapore-based-caregiving-startup-homage-raises-30m-series-c/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAJPl9ewGP8Q6BDiQ3gAKTFqtucPF7IHWeLvvCbsr5rVm3K_pB70zbBssEOXan2VfI5TTFN2q8vbj_qcchBqjO3zEyRB_XEJ8sfzTjD8f2RX0qIIKJPHrO7NhV65xgjV4YEtOL_LRKVC2KPvfG6ycxATxOE3u9_hKEqMtiv-Zh8XF'
#article(test_url)

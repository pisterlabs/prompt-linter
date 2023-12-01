import os
import shutil
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter #using text splitter to split the documents into chunks
from langchain.embeddings import OpenAIEmbeddings  # importing the embeddings
from langchain.vectorstores import Chroma # importing chroma db for vector stores
from langchain.chat_models import ChatOpenAI # loading the llm model
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA


api_key = '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'
HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}

def filter_social_media_links(links):
    excluded_domains = ['www.facebook.com', 'www.twitter.com', 'www.instagram.com', 'www.youtube.com', 'www.linkedin.com']  # we can add more social media domains as needed
    filtered_links = []

    for link in links:
        parsed_url = urlparse(link)
        domain = parsed_url.netloc.lower()

        if not any(domain in ex_domain for ex_domain in excluded_domains):
            filtered_links.append(link)

    return filtered_links

def get_links(url):
    a_links = []
    r = requests.get(url, headers=HEADERS, verify=False)
    soup = BeautifulSoup(r.content, 'html.parser')
    # print(soup.prettify())
    # title = soup.title.string.strip()
    # print(title)
    links = soup.find_all('a')
    for link in links:
        if link.get('href') == None:
            continue
        if link.get('href').startswith('http'):
            a_links.append(link.get('href'))
    return a_links
    
def get_para(link):
    paragraphs = []
    r = requests.get(link, headers=HEADERS, verify=False)
    soup = BeautifulSoup(r.content, 'html.parser')
    paras = soup.find_all('p')
    for para in paras:
        paragraphs.append(para.text.strip())

    return paragraphs

def get_text(path):
    with open(path, 'r') as f:
        text = f.read()
    return text

def write_para():
    if os.path.exists('data'):
        shutil.rmtree('data')

    os.mkdir('data')

    path = os.listdir('text/')
    para = ""

    for p in path:
        path = 'text/' + p

        with open(path, 'r') as f:
            text = f.read()
            para = para + '\n' +str(text)
            f.close()

    f = open("data/para.txt", "w")
    f.write(para)
    f.close()

# fuction to load the documents
def load_documents(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()

    return documents

#fuction to split the documents
def split_documents(documents, chunk_size=1000, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    return chunks

# function to merge scraped text data from the web
def merge_data(model, company_names):

    if model == 'langchain':
        path = 'Langchain'
        if os.path.exists(path):
            shutil.rmtree(path)

        os.mkdir('Langchain/')
        os.mkdir('Langchain/data/')

        for company in company_names:
            company_path = company + '/'
            # path = 'langchain_answers'
            para_dir = 'Langchain/data/'+ company_path + 'para.txt'
            os.mkdir('Langchain/data/'+ company_path)

            path = os.listdir('raw_text/' + company_path)
            para = ""

            # extracting the text from txt files and merging them into one file
            for p in path:
                path = 'raw_text/' + company_path + p

                with open(path, 'r') as f:
                    text = f.read()
                    para = para + '\n' +str(text)
                    f.close()

            f = open(para_dir, "w")
            f.write(para)
            f.close()

    elif model == 'valhalla':
        path = 'Valhalla'
        if os.path.exists(path):
            shutil.rmtree(path)

        os.mkdir('Valhalla/')
        os.mkdir('Valhalla/data/')
        for company in company_names:
            company_path = company + '/'
            # path = 'langchain_answers'
            para_dir = 'Valhalla/data/'+ company_path + 'para.txt'
            os.mkdir('Valhalla/data/'+ company_path)

            path = os.listdir('raw_text/' + company_path)
            para = ""

            # extracting the text from txt files and merging them into one file
            for p in path:
                path = 'raw_text/' + company_path + p

                with open(path, 'r') as f:
                    text = f.read()
                    para = para + '\n' +str(text)
                    f.close()

            f = open(para_dir, "w")
            f.write(para)
            f.close()


def llm_ans(docs, questions):

    os.environ['OPENAI_API_KEY'] = api_key
    embeddings = OpenAIEmbeddings()

    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)

    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type='stuff')

    # chain = load_qa_chain(llm, chain_type='stuff', verbose=True)

    query = f"Based on the given context, answers all the questions given {questions} in bullet format."
    # matching_docs = db.similarity_search(query)
    # answer = chain.run(input_documents=matching_docs, question=query)
    answer = qa.run(query)

    return answer

def get_relevant_docs(docs, query, embeddings):
    db = Chroma.from_documents(docs, embeddings)
    docs = db.max_marginal_relevance_search(query, k=5)
    context = "".join([doc.page_content for doc in docs])

    return context
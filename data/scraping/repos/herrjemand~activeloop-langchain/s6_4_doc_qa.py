from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

import requests
from newspaper import Article
import time

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
import chromadb

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
}

article_urls = [
    "https://www.artificialintelligence-news.com/2023/05/16/openai-ceo-ai-regulation-is-essential/",
    "https://www.artificialintelligence-news.com/2023/05/15/jay-migliaccio-ibm-watson-on-leveraging-ai-to-improve-productivity/",
    "https://www.artificialintelligence-news.com/2023/05/15/iurii-milovanov-softserve-how-ai-ml-is-helping-boost-innovation-and-personalisation/",
    "https://www.artificialintelligence-news.com/2023/05/11/ai-and-big-data-expo-north-america-begins-in-less-than-one-week/",
    "https://www.artificialintelligence-news.com/2023/05/02/ai-godfather-warns-dangers-and-quits-google/",
    "https://www.artificialintelligence-news.com/2023/04/28/palantir-demos-how-ai-can-used-military/"
]

session = requests.Session()
pages_content = [] # where we save the scraped articles

for url in article_urls:
    try:
        time.sleep(2) # sleep two seconds for gentle scraping
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            article = Article(url)
            article.download() # download HTML of webpage
            article.parse() # parse HTML to extract the article text
            pages_content.append({ "url": url, "text": article.text })
        else:
            print(f"Failed to fetch article at {url}")
    except Exception as e:
        print(f"Error occurred while fetching article at {url}: {e}")


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

chroma_dataset_name = "assistant_64"
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
vecdb = Chroma(
    client=chroma_client,
    collection_name=chroma_dataset_name,
    embedding_function=embeddings,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

all_texts = []
all_metadatas = []

for d in pages_content:
    chunks = text_splitter.split_documents(d["text"])
    for chunk in chunks:
        all_texts.append(chunk)
        all_metadatas.append({"source": d["url"]})

vecdb.add_texts(all_texts, all_metadatas)

llm = OpenAI(model_name="text-davinci-003", temperature=0.0)
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vecdb.as_retriever(),
)

d_response = chain({"question": "What does Geoffrey Hinton think about recent trends in AI?"})

print("Response:")
print(d_response["answer"])
print("Sources:")
for source in d_response["sources"].split(", "):
    print("- " + source)
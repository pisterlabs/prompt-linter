import requests
import threading
from flask import jsonify
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.schema import Document
import os
from dotenv import load_dotenv
import json

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class APIRequestThread(threading.Thread):
    def __init__(self, url):
        self.data = None
        self.url = url
        threading.Thread.__init__(self)

    def run(self):
        response = requests.get(self.url)
        self.data = response.json()


def run_scraper(amazon, walmart):
    print(amazon)
    print(walmart)
    url1 = f'http://127.0.0.1:9080/crawl.json?spider_name=amazon_reviews&start_requests=true&crawl_args={{"asin": "{amazon}"}}'
    url2 = f'http://127.0.0.1:9080/crawl.json?spider_name=wallmart_reviews&start_requests=true&crawl_args={{"asin": "{walmart}"}}'

    thread1 = APIRequestThread(url1)
    thread2 = APIRequestThread(url2)

    # Start both threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    # Combine data from both threads
    combined_data = {
        'data1': thread1.data,
        'data2': thread2.data,
    }
    combined_data = combined_data['data1']['items'] + combined_data['data2']['items']
    print(len(combined_data))
    return openaiRequest(jsonify(combined_data))


def openaiRequest(data):
    data_json = data.get_json()  # Extract the JSON data from the Response object
    product = ''
    product_flag = False
    documents = []
    for item in data_json:
        metadata = {'rating': item['rating']}
        if item['title'] is not None:
            metadata['title'] = item['title']
        if item['product'] is not None:
            metadata['product'] = item['product']
            if product_flag is False:
                product = item['product']
                product_flag = True

        document = Document(page_content=item['text'], metadata=metadata)
        documents.append(document)

    # Split the text in chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Create a persistent, file-based vector store
    directory = 'index_store'
    vector_index = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=directory)
    vector_index.persist()

    # Create the retriever and the query-interface
    retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    qa_interface = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=retriever,
                                               return_source_documents=True)

    # Query GPT-3
    response = qa_interface("""Analyze only the following collection of reviews and employ topic modeling techniques to categorize the feedback into specific features of the product.    
    Divide each feature in positive characteristics and in negative characteristics.
    Response format provided in a json format like this: {Features:[{
                    -name: x
                    -Positive Reviews:(full reviews only the ones about this feature)
                    -Negative Reviews:(full reviews only the ones about this feature)
                    }]}

    Do not repeat the same review twice.
    If there are no positive or negative characteristics, write "Not applicable".
    Give at least 6 Features.
    The product is: """ + product + """
    Provide it in JSON format.""")
    # Convert JSON to Python dictionary
    json_data = json.loads(response['result'])
    return {"product_name": product, "features": convert_structure(json_data)}


def convert_structure(data):
    converted_data = {}

    for feature in data["Features"]:
        feature_name = feature["name"]
        positive_reviews = feature["Positive Reviews"]
        negative_reviews = feature["Negative Reviews"]

        converted_data[feature_name] = {
            "name": feature_name,
            "positive_reviews": positive_reviews,
            "negative_reviews": negative_reviews
        }

    return converted_data

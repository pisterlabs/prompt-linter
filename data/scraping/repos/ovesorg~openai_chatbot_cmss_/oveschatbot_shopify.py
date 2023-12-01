import json
import pandas as pd
import requests
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import NotionDBLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.retrievers import TFIDFRetriever
from getpass import getpass
import os
url = "https://oves-2022.myshopify.com/api/2023-04/graphql.json"
page =1
def get_json(url, page):
    try:
        print('shopify password')
        url = "https://oves-2022.myshopify.com/api/2023-04/graphql.json"
        payload = json.dumps({
          "query": "query Product($first: Int) {\n  products(first: $first) {\n    nodes {\n      id\n      title\n      handle\n      descriptionHtml\n      publishedAt\n      createdAt\n      updatedAt\n      vendor\n      tags\n      createdAt\n      \n    }\n  }\n}",
          "variables": {
            "first": 100
          }
        })


        payloads = "{\"query\":\"query MyQuery {\\r\\n  articles(first: 100) {\\r\\n    nodes {\\r\\n      content\\r\\n    }\\r\\n  }\\r\\n}\",\"variables\":{}}"
        headers = {
          'Content-Type': 'application/json',
          'X-Shopify-Storefront-Access-Token': getpass(),
          'X-Shopify-Api-Version': '2023-04'
        }

        responses = requests.request("POST", url, headers=headers, data=payloads)
        products_dicts = json.loads(responses.text[20:-2])

        OUTPUT_DIR= os.getcwd()
        f = open(os.path.join(OUTPUT_DIR, 'articles.csv'), 'w')
        f.write('')
        df = pd.DataFrame.from_dict(products_dicts['nodes'])
        df.to_csv("articles.csv",index = True, header = True,encoding="utf8")
        response = requests.request("POST", url, headers=headers, data=payload)
        products_json = response.text[20:-2]
        products_dict = json.loads(products_json)
        #print(products_dict)
        OUTPUT_DIR= os.getcwd()
        f = open(os.path.join(OUTPUT_DIR, 'shopifyy_data.csv'), 'w')
        f.write('')
        df = pd.DataFrame.from_dict(products_dict['nodes'])
        df.to_csv("shopifyy_data.csv",index = True, header = True,encoding="utf8")

    except requests.exceptions.HTTPError as error_http:
        print("HTTP Error:", error_http)

    except requests.exceptions.ConnectionError as error_connection:
        print("Connection Error:", error_connection)

    except requests.exceptions.Timeout as error_timeout:
        print("Timeout Error:", error_timeout)

    except requests.exceptions.RequestException as error:
        print("Error: ", error)

get_json(url, page)
loader = DirectoryLoader(os.getcwd(), glob="**/*.csv")
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)
print('openai_secret_key')
h = getpass()
embeddings = OpenAIEmbeddings(openai_api_key=h)
db = Chroma.from_documents(docs, embeddings)
llm = OpenAI(temperature=0.8, openai_api_key=h)
retriever = db.as_retriever()
history = [{'customer':'你好','ovsmart':'你好，有什么可以帮到你的吗？'},{'customer':'hello','ovsmart':'Hi there, how can I help you?'}]
template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question and answer starting with Ovsmart then full colon and the response: 
incase of maximum token limit tell the user to be specific and dont through in an error message. if the customer asks in chinese, ovsmart should respond in chinese. if the customers asks in english ovsmart must respond in english
Be creative enough and keep conversational history to have humanly conversation.  Give short and correct answers based on the the content given. 
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    verbose=0,
    chain_type_kwargs={
        "verbose": False,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)
while True:
    print(qa.run({"query": input('\n'"customer:")}))

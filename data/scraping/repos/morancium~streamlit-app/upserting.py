# in this script we will try to extract all the info from the website then upsert it into chroma db
# we will use the terminal to configure the input style
import argparse
import os
import json
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from config import OPENAI_API_KEY
import shutil
import time


parser = argparse.ArgumentParser(
                        prog='upserting',
                        description='provide URL to scrape the data',
                        epilog='either provide the specific URL or base URL')
parser.add_argument("-r", "--reset",help="To refresh the whole database",action='store_true',default=False)
parser.add_argument("-u", "--url",help="Upsert from a set of URLs",nargs="+",type=str)
parser.add_argument("-del", "--delete",help="To delete the whole database",action='store_true',default=False)
parser.add_argument("-pd", "--persist_directory",help="Relative path to the persist_directory",type=str, default="dbdb")
args = parser.parse_args()

refresh=args.reset
urls=args.url
delete=args.delete

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
persist_directory="dbdb"

def deleting():
    if(os.path.exists(persist_directory)):
                file=os.listdir(persist_directory)
                for i in file:
                    if(os.path.isdir(os.path.join(persist_directory,i))):
                        innerfile=os.listdir(os.path.join(persist_directory,i))
                        for j in innerfile:
                            os.remove(os.path.join(persist_directory,i,j))
                    else:
                        os.remove(os.path.join(persist_directory,i))
                        # shutil.rmtree('dbdb/index')
    else:
            print("Database already dosent exist")


def reset():
    animation = "|/-\\"
    idx = 0
    url = "https://docs.vitaracharts.com/guideMapFeatures/defaultMaps.html"
    landing="https://docs.vitaracharts.com/guideAllCharts/about.html"
    base="https://docs.vitaracharts.com"
    tab_links=[]
    response = requests.get(landing)
    soup = BeautifulSoup(response.content, "lxml")
    print("Scraping fresh data...")
    # tabs= soup.find_all("nav",class_="nav nav-tabs")
    # tab_links_outer=tabs[0].find_all("a",class_="nav-item nav-link",href=True)
    # for i in tab_links_outer:
    #     tab_links.append(base+i['href'])
    # print(tab_links)
    tab_links=["https://docs.vitaracharts.com/guideAllCharts/about.html","https://docs.vitaracharts.com/guideGridFeatures/appearance.html","https://docs.vitaracharts.com/guideIBCSCommonFeatures/about.html","https://docs.vitaracharts.com/guideMapFeatures/defaultMaps.html","https://docs.vitaracharts.com/customization/about.html","https://docs.vitaracharts.com/customization/about.html"]
    Meta_texts=[]
    Meta_json=[]
    for url in tab_links:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        urls=[]
        title=[]
        soup = BeautifulSoup(response.content, "lxml")
        tags= soup.find_all("nav",class_="nav nav-pills leftMenu")
        for i in range(len(tags)):
            m=tags[i].find_all("a",href=True)
            for j in range(len(m)):
                title.append(m[j].text)
                # print((m[j].text))
                urls.append(base+m[j]['href'])
        # print(len(urls))
        # print(len(title))
        for i, url in enumerate(urls):
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "lxml")
            texts=soup.find_all("div",class_="col-md-9 rightContent")
            body=texts[0].text
            body=body.replace('\n','')
            body=body.replace('\u2018','\'')
            body=body.replace('\u2019','\'')
            body=body.replace('\t','')
            body=body.replace('\u00a0','')
            body=body.replace('\u201c','')
            body=body.replace('\u201d','')
            Meta_json.append({
                "title":title[i],
                "body":body,
                "url":url
            })
            Meta_texts.append(body)
            # break
        # json_data = json.dumps(Meta_json, indent=4)
        # with open('output.json', 'w') as outfile:
        #     json.dump(Meta_json, outfile)
    print("Storing into the vectorstore, this might take a while...")
    for i in Meta_json:
        # Chunking
        doc=i["body"]
        texts=text_splitter.split_text(doc)
        for t in texts:
            print(animation[idx % len(animation)], end="\r")
            # time.sleep(.2)
            idx += 1
            vectordb=Chroma.from_texts([t], embedding=embeddings,persist_directory=persist_directory,metadatas={"source":i["url"]})
            vectordb.persist()
            vectordb = None

if(delete):
        deleting()

# Now scraping a particular set of Urls
if (urls):
    Meta_texts=[]
    Meta_json=[]
    print(urls)
    for i, url in enumerate(urls):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "xml")
        texts=soup.find_all("div",class_="col-md-9 rightContent")
        body=texts[0].text
        body=body.replace('\n','')
        body=body.replace('\u2018','\'')
        body=body.replace('\u2019','\'')
        body=body.replace('\t','')
        body=body.replace('\u00a0','')
        body=body.replace('\u201c','')
        body=body.replace('\u201d','')
        Meta_json.append({
            # "title":title[i],
            "body":body,
            "url":url
        })
        Meta_texts.append(body)
        # break
    json_data = json.dumps(Meta_json, indent=4)
    # with open('output.json', 'w') as outfile:
    #     json.dump(Meta_json, outfile)
    
    for i in Meta_json:
        # Chunking
        doc=i["body"]
        texts=text_splitter.split_text(doc)
        for t in texts:
            vectordb=Chroma.from_texts([t], embedding=embeddings,persist_directory=persist_directory,metadatas={"source":i["url"]})
            vectordb.persist()
            vectordb = None

if(refresh):
    start=time.time()
    deleting()
    reset()
    end=time.time()
    print("Total time taken:", end-start,"s")
    print("DONE!!")
    pass
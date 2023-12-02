#Ingest all the wiki
import requests
from bs4 import BeautifulSoup
import hashlib
from langchain.docstore.document import Document
import time
from . import indexbuilder

class Wiki(indexbuilder.IndexBuilder):
    def __init__(self, config,options, baseUrl, entryPoint, malformGuard):
        super().__init__(config,options)
        self.index=[]
        self.baseUrl=baseUrl
        self.entryPoint=entryPoint
        self.malformGuard=malformGuard
        self.rebuildIndex(self.baseUrl+"/"+self.entryPoint)

    def rebuildIndex(self, url):
        # Todo: Might want to run this recursively
        content = requests.get(url).content
        soup = BeautifulSoup(content, 'html.parser')
        def addLink(link):
            href=link['href']            
            if not "://" in href: #Relative?
                if not href.startswith("/"):
                    href="/"+href
                parentUrl=url.rstrip("/")
                # substring
                parentUrl=parentUrl[:parentUrl.rfind("/")]
                
                href=parentUrl+href

            if href.startswith(self.baseUrl):
               self.index.append(href)
        
        for link in soup.select('.nav-link'):
            addLink(link)

        for link in soup.select('.wiki-rightbar a'):
            addLink(link)

            


    def __iter__(self):
        for link in self.index:
            for i in range(0, 10):
                try:
                    print("Fetch", link)
                    req = requests.get(link)
                    # if req error code, throw exception
                    req.raise_for_status()
                    content = req.content
                    if not self.malformGuard in content.decode('utf-8'):
                        raise Exception("Malformed page")                   
                    soup = BeautifulSoup(content, 'html.parser')
                    articlesFull=""
                    for article in soup.select('article'):
                        text =article.get_text()
                        articlesFull+="\n"+text

                    for article in soup.select('#wiki-body'):
                        text =article.get_text()
                        articlesFull+="\n"+text

                    articlesFull = "\n".join([t for t in articlesFull.split("\n") if t])
                    hash=self._getDocId(content)
                    doc = Document(page_content=articlesFull, metadata={"source": link, "hash":hash})
                    yield doc
                    break
                except Exception as e:
                    print("Error", link, e)
                    time.sleep(1)
           
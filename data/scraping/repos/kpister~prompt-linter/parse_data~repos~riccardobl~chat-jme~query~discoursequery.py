
import os,json
import hashlib
from langchain.docstore.document import Document
import requests
import markdownify
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from bs4 import BeautifulSoup
from embeddings import EmbeddingsManager
from . import basequery
import gc
import urllib
import multiprocessing
import utils
from Summary import Summary

# This contains several Ugly hacks to contain memory usage.
# Needs a rewrite!
class DiscourseQuery( basequery.BaseQuery):
    def __init__(self, config,url, searchFilter="in:first order:likes", knowledgeCutoff="2023-02-03",apiKey=None, apiSecret=None):
        self.CONFIG = config
        self.url = url
        self.searchFilter=searchFilter
        self.knowledgeCutoff=knowledgeCutoff


    def _createFragments(self,topicId,content,link):
        content = "\n".join([t for t in content.split("\n") if t])
        hash=hashlib.sha256(link.encode('utf-8')).hexdigest()    
        doc = Document(page_content=content, metadata={"source": link, "hash":hash})

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=512,
            chunk_overlap=0,
            length_function=len,
        )
        frags=[]
        i=0
        for chunk in splitter.split_text(doc.page_content):
            doc=Document(page_content=chunk, metadata=doc.metadata)
            v=EmbeddingsManager.new(doc,self.CONFIG["DEVICE"])
            frags.append(v)
        return frags



    def _parseTopic(self,topicId, maxNumReplies=5):
        discourseUrl=self.url
        url = f"{discourseUrl}/t/{topicId}.json"
        cachePath=self._getCachePath(topicId)

        d=None
        def getData():
            nonlocal d
            if d!=None: return d
            print("Fetch",url)
            headers = {    }
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception("Error fetching topic "+topicId)
            d=response.json()
            return d

        def getV():
            questionPath=os.path.join(cachePath,"question.binZ")
            if os.path.exists(questionPath):
                return EmbeddingsManager.read(questionPath)
            else:
                print("Get initial question of",topicId)
                data=getData()
                initialQuestion=data["title"]+"\n"+data["post_stream"]["posts"][0]["cooked"]
                initialQuestion=Summary.summarizeHTML(initialQuestion,max_length=256,min_length=64,withCodeBlocks=True)
                #print("Question:",initialQuestion)
                v=EmbeddingsManager.new(Document(page_content=initialQuestion),self.CONFIG["DEVICE"])
                EmbeddingsManager.write(questionPath,v)
                return v

        def getContent():
            contentPath=os.path.join(cachePath,"fragments.binZ")
            if os.path.exists(contentPath):
                return EmbeddingsManager.read(contentPath)
            else:
                data=getData()
                print("Process",topicId)
                content=[]
                contentPart=""
                isQuestion=True
                isFirst=True
                topicAuthorId=data["user_id"]
                posts = data["post_stream"]["posts"]
                def flush():
                    nonlocal contentPart
                    nonlocal isQuestion
                    nonlocal isFirst
                    if len(contentPart)==0: return
                    c=""
                    if isQuestion:
                        c+="\n\nQUESTION:\n"
                        if isFirst:
                            author=data["post_stream"]["posts"][0]["name"]
                            if author==None: author=data["post_stream"]["posts"][0]["username"]
                            c+=data["title"]+"\n"+"Author: "+author+"\n"  
                            isFirst=False
                    else:
                        c+="\n\nANSWER:\n"                
                    #c+=contentPart
                    c+=Summary.summarizeHTML(contentPart,f"{discourseUrl}/t/{topicId}",max_length=256,min_length=64,withCodeBlocks=not isQuestion)
                    contentPart=""    
                    #print("Content",c)
                    content.append(c)                    
                for post in posts:
                    postAuthorId=post["user_id"]
                    postText=post["cooked"]                
                    if isQuestion and postAuthorId!=topicAuthorId:
                        flush()
                        isQuestion=False
                    elif not isQuestion and postAuthorId==topicAuthorId:
                        flush()
                        isQuestion=True     
                    contentPart+=postText+"\n"
                flush()

                if len(content)>maxNumReplies:
                    content=content[:1]+content[-maxNumReplies:]
                content="\n".join(content)

                content=Summary.summarizeHTML(content,f"{discourseUrl}/t/{topicId}",max_length=512,min_length=120,withCodeBlocks=True)
                content = markdownify.markdownify(content, heading_style="ATX",autolinks=True,escape_asterisks=False,escape_underscores=False)
                content = self._createFragments(topicId, content,discourseUrl+"/t/"+str(topicId))
                EmbeddingsManager.write(contentPath,content)
                return content

        return {
            "id":topicId,
            "frags":getContent,
            "v":getV
        }

    def _getCachePath(self,id):
        urlHash=hashlib.sha256(self.url.encode('utf-8')).hexdigest()
        cacheRoot=os.path.join(self.CONFIG["CACHE_PATH"],"discourse",urlHash)
        cachePath=os.path.join(cacheRoot,str(id))
        if not os.path.exists(cachePath):
            os.makedirs(cachePath)
        return cachePath
 
    def _search(self, searchTerms, question,searchLimit=1,maxTopicsToSelect=1,maxFragmentsToReturn=3,maxNumReplies=2, merge=False):
        discourseUrl=self.url


        # Search
        def getTopics(term):
            termTopics=[]
            def search():    
                params = {
                    "q": term+" "+self.searchFilter+" before:"+self.knowledgeCutoff
                }        
                print("searching",discourseUrl, params)
                response = requests.get(discourseUrl+"/search.json", params=params)
                if response.status_code != 200:
                    print("Error searching discourse")
                    raise Exception("Error searching discourse")
                jsonData=response.json()        
                return jsonData

            try:
                jsonData= utils.retry(search,3,1)
                if not "topics" in jsonData: return []
                for topic in jsonData["topics"]:
                    if len(termTopics)>=searchLimit: break
                    id=topic["id"]
                    topicData=self._parseTopic(id,maxNumReplies)
                    termTopics.append(topicData)
            except Exception as e:
                print("Error searching discourse",e)
            return termTopics

        topics=[]
        for term in searchTerms:
            topics.extend(getTopics(term))
        
        cache={}
        
        
        #for topic in topics:
        def assignScore(topic):
            v=topic["v"]
            res=EmbeddingsManager.queryIndex(v(),question, k=1, cache=cache, group=EmbeddingsManager.GROUP_GPU)
            score=None
            for rdoc in res:
                rscore=rdoc[1]
                if not score or rscore<score:
                    score=rscore
            topic["score"]=score
            return topic
        
        for topic in topics:
            assignScore(topic)

        topics = sorted(topics, key=lambda x: x["score"], reverse=False)[:maxTopicsToSelect]

        gc.collect()

        fragments=[]
        for t in topics:
            fragments.extend(t["frags"]())            
        topics=EmbeddingsManager.query(fragments,question, k=3,n=maxFragmentsToReturn, cache=cache, group=EmbeddingsManager.GROUP_GPU)           
        if merge:
            print("Found",len(topics),"topics, Merge")        
            mergedTopic=""
            for t in topics:
                mergedTopic+=t.page_content+"\n"
            mergedTopic=Summary.summarizeHTML(mergedTopic,min_length=100,max_length=400,withCodeBlocks=True)
            print("Merged in ",len(mergedTopic),"chars")
            topics= [Document(page_content=mergedTopic, metadata={"source": f"{discourseUrl}/search", "hash":""})]
        return topics

    def getAffineDocs(self, question, context, keywords, shortQuestion,  wordSalad=None, unitFilter=None,
            maxFragmentsToReturn=3, maxFragmentsToSelect=6, merge=False):
        seachTerms=[]
        #seachTerms.append(question)
        seachTerms.extend(keywords)
        seachTerms=seachTerms[:3]
        #return self._search(seachTerms,question)        
        return  self._search(seachTerms,question)


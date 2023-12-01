from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from bs4 import BeautifulSoup
import gc
import mistune
import markdownify
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import yake
from transformers import pipeline
import threading
import queue
from urllib.parse import urljoin


zz = OpenAI()
class Summary:
   
    summarizer=None
    summarizer2=None
    tokenizer2=None
    parser=None
    CONFIG=None
    tokenizer=None

    summarizerI=0
    summarizerLock=threading.Lock()

    @staticmethod
    def init(CONFIG):
        Summary.useGPU=CONFIG.get("DEVICE","cpu")=="gpu" or CONFIG.get("DEVICE","cpu")=="cuda"
        Summary.CONFIG=CONFIG
        Summary.useSumy=CONFIG.get("USE_SUMY",False)
        if not Summary.useSumy:
            if Summary.summarizer==None:
                print("Preloading flan-t5-base-samsum")
                parallel=1
                Summary.summarizer = [
                    
                        pipeline("summarization", model='philschmid/flan-t5-base-samsum', device=0 if Summary.useGPU else -1)
                    for i in range(0,parallel)
                ]
                print("Done")
        LANGUAGE="english"
        stemmer = Stemmer(LANGUAGE)
        Summary.summarizer2 = Summarizer(stemmer)
        Summary.summarizer2.stop_words = get_stop_words(LANGUAGE)
        Summary.tokenizer2=Tokenizer(LANGUAGE)


    @staticmethod
    def getKeywords(content,n=5):
        language = "en"
        max_ngram_size = 3
        deduplication_threshold = 0.9
        numOfKeywords = n
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
        keywords = custom_kw_extractor.extract_keywords(content)
        return [ t[0] for t in keywords]



    @staticmethod
    def summarizeMarkdown(content,url="",min_length=10,max_length=100, withCodeBlocks=True, length=None,fast=False):
        contentLen=length
        if contentLen==None: contentLen=Summary.getLength(content,fast=fast)
        if contentLen<min_length: return content
        if max_length>contentLen: max_length=contentLen

        content = mistune.html(content)
        content=Summary.summarizeHTML(content,url,min_length,max_length,withCodeBlocks,fast=fast)
        content = markdownify.markdownify(content, heading_style="ATX",autolinks=True,escape_asterisks=False,escape_underscores=False)

    @staticmethod
    def getLength(content,fast=False):
        if Summary.useSumy or fast:
            return len(Summary.tokenizer2.to_sentences(content))
        else:
            tokenizer = Summary.summarizer[Summary.summarizerI].tokenizer
            input_ids = tokenizer.encode(content)
            return len(input_ids)

    @staticmethod
    def summarizeText(content,min_length=10,max_length=100,length=None,fast=False):       
        contentLen=length
        if contentLen==None: contentLen=Summary.getLength(content)
        if contentLen<min_length: return content
        if max_length>contentLen: max_length=contentLen

        if Summary.useSumy or fast:
            try:                
                SENTENCES_COUNT = max_length
                parser = PlaintextParser.from_string(content, Summary.tokenizer2)
                text_summary=""
                for sentence in Summary.summarizer2(parser.document, SENTENCES_COUNT):
                    text_summary+=str(sentence)
                return text_summary
            except Exception as e:
                print("Error summarizing",e)
                return ""
        else:        
            summarizer=None
            with Summary.summarizerLock:
                summarizer=Summary.summarizer[Summary.summarizerI]
                Summary.summarizerI+=1  
                if Summary.summarizerI>=len(Summary.summarizer):
                    Summary.summarizerI=0
           
            res=summarizer(content,min_length=min_length,max_length=max_length)

            return res[0]["summary_text"]

    @staticmethod
    def summarizeHTML(content,url="",min_length=10,max_length=100, withCodeBlocks=True,length=None,fast=False):
        contentLen=length
        if contentLen==None: contentLen=Summary.getLength(content,fast=fast)
        if contentLen<min_length: return content
        if max_length>contentLen: max_length=contentLen

        try:       
            # Extract links
            soup = BeautifulSoup(content, 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                url = urljoin(url, href)
                link.string = url

            # Extract code blocks
            codeBlocks=""
            cc=soup.find_all("pre")
            for c in cc:
                if withCodeBlocks:
                    i=0
                    i+=1
                    rpl=f"[{i}]"
                    codeBlocks+=rpl+" <pre><code>"
                    codeBlocks+=c.text
                    codeBlocks+="</code></pre>"            
                    c.string = rpl
                else:
                    c.string = ""
                
            # To plain text
            texts = soup.findAll(text=True)
            text_summary = u" ".join(t.strip() for t in texts)
            text_summary=Summary.summarizeText(text_summary,min_length,max_length,fast=fast)
            text_summary+=codeBlocks

            return text_summary
        except Exception as e:
            print("Error summarizing",e)
            return ""
prompt = PromptTemplate("arge")

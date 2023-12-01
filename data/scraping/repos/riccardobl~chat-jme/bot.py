import os

import utils

import traceback
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
import langchain
from langchain.cache import InMemoryCache
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory,ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from embeddings import EmbeddingsManager
from flask import Flask, send_from_directory
import json
import time
import threading
import secrets
import string
import hashlib
from flask import request
from langchain.cache import InMemoryCache,SQLiteCache
import re
import requests
from waitress import serve
from translator import Translator
import sys
from query.discoursequery import DiscourseQuery
from query.embeddingsquery import EmbeddingsQuery
from Summary import Summary
import uuid
from langchain.llms import NLPCloud
from langchain.llms import AI21
from langchain.llms import Cohere
from SmartCache import SmartCache
CONFIG=None
QUERIERS=[]

args=sys.argv
confiFile=args[1] if len(args)>1 else "config.json"
print("Use config file", confiFile)
with open(confiFile, "r") as f:
    CONFIG=json.load(f)
    EmbeddingsManager.init(CONFIG)
    Summary.init(CONFIG)
    QUERIERS=[
        EmbeddingsQuery(CONFIG),
        DiscourseQuery(
            CONFIG,CONFIG["JME_HUB_URL"],
            searchFilter=CONFIG["JME_HUB_SEARCH_FILTER"],
            knowledgeCutoff=CONFIG["JME_HUB_KNOWLEDGE_CUTOFF"]
        )
    ]
    Translator.init(CONFIG)

def getAffineDocs(question,context,keywords,shortQuestion, wordSalad=None, unitFilter=None,
    maxFragmentsToReturn=3, maxFragmentsToSelect=12,merge=False):
    affineDocs=[]

   
    for q in QUERIERS:
        print("Get affine docs from",q,"using question",question,"with context",context,"and keywords",keywords)
        t=time.time()
        v=q.getAffineDocs(
            question, context, keywords,shortQuestion, wordSalad, unitFilter,
            maxFragmentsToReturn=maxFragmentsToReturn,
            maxFragmentsToSelect=maxFragmentsToSelect,
            merge=merge        
        )
        print("Completed in",time.time()-t,"seconds.")
        if v!=None:
            affineDocs.extend(v)
    return affineDocs
    
def rewriteError(error):
    if error.startswith("Rate limit reached ") :
        return "Rate limit."

def rewrite(question):
    # replace app, applet, game, application with simple application 
    question=re.sub(r"\b(app|applet|game|application)\b", "simple application", question, flags=re.IGNORECASE)

    return question



def createChain():
    



    # Backward compatibility
    model_name=CONFIG.get("OPENAI_MODEL","text-davinci-003")
    llm_name="openai"
    ######## 
    
    llmx=CONFIG.get("LLM_MODEL",None) # "openai:text-davinci-003" "cohere:xlarge"
    if llmx!=None: 
        if ":" in llmx:
            llm_name,model_name=llmx.split(":")
        else:
            llm_name,model_name=llmx.split(".")



    template = ""
    template_path="prompts/"+llm_name+"."+model_name+".txt"
    if not os.path.exists(template_path):
        template_path="prompts/openai.text-davinci-003.txt"
    
    with open(template_path, "r") as f:
        template=f.read()

    prompt = PromptTemplate(
        input_variables=[ "history", "question", "summaries"], 
        template=template
    )

    llm=None
    history_length=700
    if llm_name=="openai":
        max_tokens=512
        temperature=0.0
        if model_name=="text-davinci-003":
            max_tokens=512
        elif model_name=="code-davinci-002":
            max_tokens=1024
            #history_length=1024            
        llm=OpenAI(
            temperature=temperature,
            model_name=model_name,
            max_tokens=max_tokens,
        )
    elif llm_name=="cohere":
        llm=Cohere(
            model=model_name,
            max_tokens=700
        ) 
        history_length=200
    elif llm_name=="ai21":
        llm=AI21(
            temperature=0.7,
            model=model_name,
        )   
    elif llm_name=="nlpcloud":
        llm=NLPCloud(
            model_name=model_name,
        )
    else:
        raise Exception("Unknown LLM "+llm_name)

    print("Use model ",model_name,"from",llm_name)

    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=history_length,human_prefix="QUESTION",ai_prefix="ANSWER", memory_key="history", input_key="question")
    chain = load_qa_with_sources_chain(
        llm,
        memory=memory, 
        prompt=prompt, 
        verbose=True,
    )

    return chain


    
def extractQuestionData(question,wordSalad):
    shortQuestion=Summary.summarizeMarkdown(question,min_length=100,max_length=1024,withCodeBlocks=False)

    context=Summary.summarizeText(wordSalad,min_length=20,max_length=32)
    keywords=[]
    keywords.extend(Summary.getKeywords(shortQuestion,2))
    keywords.extend(Summary.getKeywords(Summary.summarizeText(wordSalad,min_length=10,max_length=20),3))

    return [question,shortQuestion,context,keywords,wordSalad]


def queryChain(chain,question):
    wordSalad=""
    for h in chain.memory.buffer: wordSalad+=h+" "
    wordSalad+=" "+question    
    [question,shortQuestion,context,keywords,wordSalad]=utils.enqueue(lambda :extractQuestionData(question,wordSalad))
    affineDocs=utils.enqueue(lambda :getAffineDocs(question,context,keywords,shortQuestion,wordSalad))
    print("Found ",len(affineDocs), " affine docs")       
    print("Q: ", shortQuestion)
    output=chain({"input_documents": affineDocs, "question": shortQuestion}, return_only_outputs=True)    
    print("A :",output)
    return output


sessions={}
langchain.llm_cache = SmartCache(CONFIG)#SQLiteCache(database_path=CONFIG["CACHE_PATH"]+"/langchain.db")

def clearSessions():
    while True:
        time.sleep(60*5)
        for session in sessions:
            if sessions[session]["timeout"] < time.time():
                del sessions[session]
threading.Thread(target=clearSessions).start()

def createSessionSecret():
    hex_chars = string.hexdigits
    timeHash=hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()[:12]
    return ''.join(secrets.choice(hex_chars) for i in range(64))+timeHash


app = Flask(__name__)    

@app.route("/langs")
def langs():
    return json.dumps(Translator.getLangs())

@app.route("/session",methods = ['POST'])
def session():
    body=request.get_json()
    lang=body["lang"] if "lang" in body  else "en"
    if lang=="auto":
        lang="en"


    if not "sessionSecret" in body or body["sessionSecret"].strip()=="":
        sessionSecret=createSessionSecret()
    else:
        sessionSecret=body["sessionSecret"]

    if sessionSecret not in sessions:
        sessions[sessionSecret]={
            "chain": createChain(),
            "timeout": time.time()+60*30
        }
    else:
        sessions[sessionSecret]["timeout"]=time.time()+60*30
    welcomeText=""
    welcomeText+=Translator.translate("en", lang,"Hi there! I'm an AI assistant for the open source game engine jMonkeyEngine. I can help you with questions related to the jMonkeyEngine source code, documentation, and other related topics.")
    welcomeText+="<br><br>"
    welcomeText+="<footer><span class=\"material-symbols-outlined\">tips_and_updates</span><span>"+Translator.translate("en", lang,"This chat bot is intended to provide helpful information, but accuracy is not guaranteed.")+"</span></footer>"

       
    return json.dumps( {
        "sessionSecret": sessionSecret,
        "helloText":Translator.translate("en",lang,"Who are you?"),
        "welcomeText":welcomeText
    })

@app.route("/query",methods = ['POST'])
def query():
    try:
        body=request.get_json()
        question=rewrite(body["question"])
        lang=body["lang"] if "lang" in body  else "en"
        
        if lang == "auto":
            lang=Translator.detect(question)

        if lang!="en":
            question=Translator.translate(lang,"en",question)

        if len(question)==0:
            raise Exception("Question is empty")
        
        sessionSecret=body["sessionSecret"]   
        
        if sessionSecret not in sessions:
            return json.dumps({"error": "Session expired"})
            
        chain=sessions[sessionSecret]["chain"]

        output=queryChain(chain,question)
       
        if lang!="en":
            output["output_text"]=Translator.translate("en",lang,output["output_text"])

        #print(chain.memory.buffer)
        return json.dumps(output)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        errorStr=str(e)
        errorStr=rewriteError(errorStr)
        return json.dumps({"error": errorStr})


@app.route('/<path:filename>')
def serveFrontend(filename):
    return send_from_directory('frontend/', filename)

@app.route('/')
def serveIndex():
    return send_from_directory('frontend/', "index.html")

@app.route('/docs', methods=['POST'])
def docs():
    body=request.get_json()
    question=body["question"]
    maxFragmentsToReturn=int(body.get("maxFragmentsToReturn",3))
    maxFragmentsToSelect=int(body.get("maxFragmentsToReturn",6))
    wordSalad=body.get("context","")+" "+question
    [question,shortQuestion,context,keywords,wordSalad]=utils.enqueue(lambda : extractQuestionData(question,wordSalad))
    affineDocs=utils.enqueue(lambda : getAffineDocs(
        question,context,keywords,shortQuestion,wordSalad,
        maxFragmentsToReturn=maxFragmentsToReturn,
        maxFragmentsToSelect=maxFragmentsToSelect
    ))
    plainDocs=[
        {
            "content":doc.page_content,
            "metadata":doc.metadata
        } for doc in affineDocs
    ]
    return json.dumps(plainDocs)


serve(app, host="0.0.0.0", port=8080, connection_limit=1000)

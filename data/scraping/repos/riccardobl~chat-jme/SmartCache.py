from langchain.cache import BaseCache
import os

import utils


from embeddings import EmbeddingsManager
import json
from typing import Any, Dict, List, Optional, Tuple

 
from langchain.schema import Generation

import time
import pickle

from Summary import Summary
import uuid

RETURN_VAL_TYPE = List[Generation]

class SmartCache(BaseCache):
    CONFIG:dict=None
    WAIT_FOR_UPDATE:dict={}

    def __init__(self,config) -> None:
        self.CONFIG= config

    def queryCache(self, shortQuestion, wordSalad,cacheConf):   
        # only last 5 lines of wordSalad
         
        CONFIG=self.CONFIG
        levels=[None]*len(cacheConf)
        for i in range(len(cacheConf)-1,-1,-1): 
            text=""
            l=cacheConf[i][0]
            if i==(len(cacheConf)-1):
                text=shortQuestion
            else:
                nextI=i+1
                text=wordSalad+" "+shortQuestion if nextI==len(cacheConf)-2 else levels[i+1][2]
                text=Summary.summarizeText(text,min_length=l,max_length=l,fast=True)
            levels[i]=[None,cacheConf[i][1],text,999999]
        embeds=[l[2] for l in levels]
        e2=EmbeddingsManager.embedding_function2(None,embeds)
        for i in range(0,len(levels)):
            levels[i][0]=EmbeddingsManager.new(levels[i][2],"gpu") # TODO: make this parallel
            levels[i][3]=e2[i]


        cachePath=os.path.join(CONFIG["CACHE_PATH"],"smartcacheV2")  
        if not os.path.exists(cachePath):
            os.makedirs(cachePath)
        for i in range(0,len(levels)):
            l=levels[i]
            isLast=i==len(levels)-1
            foundSub=False
            for f in os.listdir(cachePath):
                if not f.endswith(".bin"): continue
                embeddingPath=os.path.join(cachePath,f)
                answerPath=embeddingPath.replace(".bin",".dat")
                subPath=embeddingPath.replace(".bin","")

                embedding=EmbeddingsManager.read(embeddingPath,group=EmbeddingsManager.GROUP_GPU)
                res=EmbeddingsManager.queryIndex(embedding,l[3],k=1,group=EmbeddingsManager.GROUP_GPU)
                score=res[0][1]
                print("Score:",score,"level score",l[1])
                if score<l[1]:
                    print("Found in cache",l[2])
                    if isLast:
                        print("Return from cache")
                        if os.path.exists(answerPath):                 
                            with open(answerPath, "rb") as f:
                                answer=pickle.load(f)
                                #answer=json.load(f)
                                return [
                                    answer,
                                    lambda x: None                          
                                ]
                    else:
                        print("Go deeper")
                        cachePath=subPath
                        foundSub=True
                        break
            if not foundSub:
                f=uuid.uuid4().hex+".bin"
                embeddingPath=os.path.join(cachePath,f)
                answerPath=embeddingPath.replace(".bin",".dat")
                subPath=embeddingPath.replace(".bin","")
                if isLast:
                    print("Not in cache!")
                    def writeAnswer(answer):
                        print("Add answer to smart cache")
                        EmbeddingsManager.write(embeddingPath,l[0])
                        with open(answerPath, "wb") as f:
                            pickle.dump(answer, f)
                            #json.dump(answer, f)
                    return [
                        None,
                        writeAnswer
                    ]
                else:
                    print("Create deeper level")
                    os.mkdir(subPath)
                    cachePath=subPath
                    EmbeddingsManager.write(embeddingPath,l[0])

                
    def lookup(self, prompt: str, llm_string: str):
        shortQuestion=prompt[prompt.rfind("QUESTION:")+len("QUESTION:"):prompt.rfind("FINAL ANSWER")]

        [answer,writer] = self.queryCache(shortQuestion,prompt,self.CONFIG["SMART_CACHE"])
        if not writer is None:
            for k in list( self.WAIT_FOR_UPDATE):
                if self.WAIT_FOR_UPDATE[k][1]<time.time():
                    del self.WAIT_FOR_UPDATE[k]
            self.WAIT_FOR_UPDATE[(prompt,llm_string)]=[writer,time.time()+1000*60*15]
        return answer

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        
        [writer,timeout]=self.WAIT_FOR_UPDATE.pop((prompt,llm_string))
        if writer is not None: writer(return_val)

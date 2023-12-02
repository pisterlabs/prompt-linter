import os
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from embeddings import EmbeddingsManager
import json
import hashlib
class IndexBuilder :
    def __init__(self,config, options):
        self.options = options
        self.config=config

    def _getDocId(self,content):
        hash=""
        merged=self.options.get("merged",False)
        if merged:
            hash=self.options["unit"]
        else:
            print("Calculate hash")
            hash=hashlib.sha256(content.encode('utf-8')).hexdigest()    
        return hash

    def updateIndex(self):
        docs=[]
        if not "INDEX_PATH" in self.config:
            raise Exception("INDEX_PATH not set")
        rootPath = os.path.join(self.config["INDEX_PATH"],self.options["unit"] if "unit" in self.options else "root")
        if not os.path.exists(rootPath):
            os.makedirs(rootPath)
        infoPath = os.path.join(rootPath,"info.json")
        optionsJson=json.dumps(self.options)
        with open(infoPath,"w",encoding="utf-8") as f:
            f.write(optionsJson)

        merged=self.options.get("merged",False)

        if merged:
            identifier=self.options["unit"]
            embedPath = os.path.join(rootPath, identifier + ".bin")
            if os.path.exists(embedPath):
                print("Already processed", identifier)
                return []


        for doc in self:
            docs.append(doc)
            if not merged: self._updateIndex(rootPath,[doc], doc.metadata["hash"],doc.metadata["source"])
        if merged:
            self._updateIndex(rootPath,docs, self.options["unit"], self.options["unit"])
        return docs

        
    def _updateIndex(self,rootPath,docs, identifier,name ):    
        try:

                

            embedPath = os.path.join(rootPath, identifier + ".bin")
            if os.path.exists(embedPath):
                print("Already processed", name)
                return

            faiss=EmbeddingsManager.new(docs,backend="gpu")

            EmbeddingsManager.write(embedPath, faiss)          

            print ("Updated",  name)
        except Exception as e:
            print("Error processing",  name, e)
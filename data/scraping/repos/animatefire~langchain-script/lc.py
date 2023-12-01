#!/usr/bin/env python3

import os
import sys
from datetime import datetime

import constants
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = sys.argv[1]
type = sys.argv[2]

loader = DirectoryLoader(".", glob="*."+type)
index = VectorstoreIndexCreator().from_loaders([loader])

response = index.query(query, llm=ChatOpenAI())
print(response + '\n')

fileName = str(input("What would you like to call this file? (langchain-log.md)\n") or "langchain-log.md")
file = open(fileName, "a+")
now = datetime.now()
file.write( "\n\n" + "** " + now.strftime("%m/%d/%Y %H:%M:%S") + ": " + query + " **" + "\n\n" + response + "\n\n" )
print('File successfully written.\n')
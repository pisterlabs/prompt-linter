## main.py - this is a ChatGPT tool called "Doug"
# Doug is here to guide you. Please do not fear Doug. Doug is your friend.
# Doug always refers to themselves in third person.

import pdb

from os import environ
from sys import exit, path
from dotenv import load_dotenv
from pathlib import Path

# Import the langchain core
import langchain

from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain import PromptTemplate, LLMChain

from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

from langchain.chains import ConversationChain
from langchain.memory import \
    ConversationBufferMemory, ConversationBufferWindowMemory, CombinedMemory, ConversationSummaryBufferMemory

from langchain.vectorstores import Chroma

## Custom Module Path Injection
# inject our module path so we can resolve 'import' statements later
path.insert(0, './data/modules')

## Custom Modules
from SignalHandler  import signalHandler
from PdfLoader      import pdfLoader

print("   - - - - - - - - DOUG V1.0 : AI ASSISTANT - - - - - - - -\n")

## Initialization #############################################################
# load our .env file, which has our OPENAI_API_KEY
load_dotenv()

USE_OPENAI=True
USE_GPT4ALL=False
USE_LLAMA=False

# to get a list of models, run utils/get-openai-models.py
dougOpenAIModel=environ.get("OPENAI_LLM_MODEL")
if (dougOpenAIModel is None) and USE_OPENAI:
    print("WARN: OPENAI_LLM_MODEL not set, defaulting to 'gpt-3.5-turbo-0301'")
    dougOpenAIModel="gpt-3.5-turbo-0301"

# Choose the LLM we want to use
if USE_OPENAI:
    print("INFO: Selected OpenAI as our LLM")
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks import get_openai_callback
    from langchain.embeddings.openai import OpenAIEmbeddings
    dougLLM = ChatOpenAI(model_name=dougOpenAIModel, temperature=0.4)
    dougEmbedding = OpenAIEmbeddings(model_name="ada")
elif USE_GPT4ALL:
    print("INFO: Selected GPT4All as our LLM")
    llmLocalPath="./data/models/gpt4all-lora-quantized-new.bin"
    from langchain.llms import GPT4All
    from langchain.embeddings import LlamaCppEmbeddings
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    dougLLM = GPT4All(model=llmLocalPath, f16_kv=False, n_ctx=512, use_mlock=True, \
        n_threads=8, n_predict=1000, temp=0.1, callback_manager=callback_manager, verbose=False)
    dougEmbedding = LlamaCppEmbeddings(model_path="data/models/gpt4all-lora-quantized-new.bin")
elif USE_LLAMA:
    print("INFO: Selected llama.cpp as our LLM")
    llmLocalPath="./data/models/gpt4all-lora-quantized-new.bin"
    from langchain.llms import LlamaCpp
    from langchain.embeddings import LlamaCppEmbeddings
    dougLLM = LlamaCpp(model_path=llmLocalPath, temperature=0.7)
    dougEmbedding = LlamaCppEmbeddings(model_path="data/models/gpt4all-lora-quantized-new.bin")
else:
    print("ERROR: No LLM enabled. Please set one of USE_OPENAI, USE_GPT4ALL, or USE_LLAMA to 'True'")
    exit(1)

osSignalHandler = signalHandler()

if (environ.get("OPENAI_API_KEY") is None) and USE_OPENAI:
    print("ERROR: OpenAI API Key is not set")
    exit(1)

# initialize our chat message memory for our interactive chat session
dougWorkingMemory = ConversationBufferWindowMemory(memory_key="history", input_key="input", k=10)
dougSummaryMemory = ConversationSummaryBufferMemory(llm=dougLLM, input_key="input")

dougMainMemory = CombinedMemory(memories=[dougWorkingMemory, dougSummaryMemory])

# TODO: load saved prompt from 'yaml' file
dougPrompt = Path('./data/prompt/doug-v1.prompt').read_text()

try:
    dougTemplate = PromptTemplate(
        input_variables=["history", "input"],
        template=dougPrompt
        )
except:
    print("Unable to create template")
    pdb.set_trace()
    exit(1)

dougChain = ConversationChain(llm=dougLLM, memory=dougMainMemory, prompt=dougTemplate)

dougPersistentMemory = './data/db'
dougDB = Chroma(persist_directory=dougPersistentMemory, embedding_function=dougEmbedding)

###############################################################################
print("\n [ commands: .quit - exit this cli app, .debug - pdb console ] ")

documents = None
totalCost = 0

while True:
    osSignalHandler.reset_signal()
    userInquiry = input('\n: ')

    if userInquiry:
        cmd = userInquiry.split()[0]

        if cmd == ".exit" or cmd == ".quit":
            dougDB.persist()
            exit(0)

        if cmd == ".debug":
            pdb.set_trace()
            continue

        if cmd == ".$$$":
            if USE_OPENAI:
                print("-> $" + str(totalCost))
            else:
                print("-> Using a local LLM, the only cost is your compute")
            continue

        if cmd == ".pdf":
            if len(userInquiry.split()) > 1:
                try:
                    arg = " ".join(userInquiry.split()[1:])
                    dougPdf = pdfLoader(persistdb=dougDB, path=arg, signalHandler=osSignalHandler)

                    dougPdf.addPathToQueue(arg)
                    dougPdf.processQueue()
                    dougPdf.storeQueue()

                except Exception as error:
                    print("  ~> " + str(error))

            else:
                print("-> pdf command requires one path argument")

            continue

        if cmd == ".search":
            if len(userInquiry.split()) > 1:
                try:
                    query = " ".join(userInquiry.split()[1:])
                    retriever = dougDB.as_retriever(search_type="mmr")

                    documents = retriever.get_relevant_documents(query)
                    print("-> returned " + str(len(documents)) + " documents")

                except Exception as error:
                    print("  ~> " + str(error))
            else:
                print("-> search command requires something to search for")

            continue

        if cmd == ".docs":
            try:
                docsAvailable = len(documents)
                if docsAvailable < 1:
                    print("-> no documents found, try searching first")
                    continue

                if len(userInquiry.split()) > 1:
                    arg = int(userInquiry.split()[1])
                    if arg > docsAvailable:
                        print("-> index out of range")
                        continue
                    else:
                        print(documents[arg])
                else:
                    print("-> " + str(docsAvailable) + " documents available")

            except Exception as error:
                print("  ~> " + str(error))

            continue

        ## Main LLM Block - reach out to LLM with inquiry
        try:
            if USE_OPENAI:
                with get_openai_callback() as callBack:
                    result = dougChain.run(input=userInquiry)
                    totalCost += round(callBack.total_cost, 5)
                    print("[" + str(round(callBack.total_cost, 5)) + "] ", end="")
            else:
                result = dougChain.run(input=userInquiry)

            if result:
                print(result)
            else:
                print("ERROR: LLM did not respond")
        except Exception as error:
            print("ERROR: " + str(error))
            continue

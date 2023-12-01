#------------------------------------------------------------------------------------------------------------
# GPTFunctions based chatbot - ripping up LangChain show of horrors and going for my own implementation
#------------------------------------------------------------------------------------------------------------

import json
from openai_function_call import openai_function
from fchain import fChain,fChainVectorDB
import re

sizeBeforeCompact=3000
promptAssembly=["intro.txt","flow1.txt","fieldformatting.txt","tables.txt","workedexample1.txt"]

def LoadPrompt(fn):
    fn="prompts/"+fn
    prompt=""
    with open(fn, 'r') as file:
        prompt = file.read()
    return prompt

def AssemblePrompt(fileNames:list[str])->str:
    prompt=""
    for fn in fileNames:
        prompt=prompt+LoadPrompt(fn)
    return prompt

def ReloadLLM():
    systemprompt=AssemblePrompt(promptAssembly)
    print("------------------ System Prompt ------------------")
    print(systemprompt)
    print("--------------- End System Prompt -----------------")
    functions=[listfields,fieldsDetails]
    chatBot=fChain(systemprompt,functions,False,False,True,"c:/temp/fchain/",model_name="ft:gpt-3.5-turbo-0613:personal::7tF0ZPhe")
    return chatBot

#Does some magic to turn xl://some_link into an href hyperlink
def handleUrls(aText):
    aConvertedText=""

    #Regex to grab values between []
    regex="(?<=\[)(.*?)(?=\])"

    #Do a regext replace on aText
    aConvertedText=re.sub(regex,lambda x: "<a href='"+x.group(0)+"'>"+x.group(0)+"</a>",aText)
    return aConvertedText

#---------------------------------------------------------------------------------

#Load in JSON fragments file/vectorize embeddings/return vectorDB
def LoadEmbeddings():
    print("Creating Embeddings from config...")
    file_path='../Config/chatbotconfig.json'
    loader = fChainVectorDB(file_path=file_path)
    vectordb = loader.load()
    print("Embeddings loaded...")  
    return vectordb

#------------------------------------------------------------------------------------------------------------
# Knowledge agent functons
#------------------------------------------------------------------------------------------------------------

@openai_function
def listfields(search:str,table:str="All") -> str:
    """Lists database fields in the database by table (leave blank or put All for all tables), shorter description for when user wants to find fields"""

    global vectordb
    print("listfields ["+search+"]")
    
    collection=vectordb.get_collection("fields_list")
    if table.lower()=="all":
        results = collection.query(
            query_texts=[search],
            n_results=10,
            # where={"metadata_field": "is_equal_to_this"}, # optional filter
            # where_document={"$contains":"search_string"}  # optional filter
        )
    else:
        results = collection.query(
            query_texts=[search],
            n_results=10,
            where={"Table Name": table}, # optional filter
            # where_document={"$contains":"search_string"}  # optional filter
        )

    ret=results["documents"]
    for v in ret:
        retList=[n for n in v]
    return json.dumps(retList)


@openai_function
def fieldsDetails(search:str) -> str:
    """Gete more details on fields in the database, longer decscription with more information and specifics (like field security level)"""

    global vectordb
    print("fieldsDetails ["+search+"]")
    
    collection=vectordb.get_collection("fields_detail")
    results = collection.query(
        query_texts=[search],
        n_results=5,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )
    ret=results["documents"]
    for v in ret:
        retList=[n for n in v]
    return json.dumps(retList)


#------------------------------------------------------------------------------------------------------------
#Flask App
#------------------------------------------------------------------------------------------------------------

import flask
from flask import Flask, request, jsonify
#from flask_cors import CORS

#App handler
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
#CORS(app)

#Load embeddings and create DB
vectordb=LoadEmbeddings()

#Load vectorstore and init chat engine
chatBot=ReloadLLM()

#Simple homepage
@app.route('/')
def index():
    return flask.render_template('main.html')

#Simple post/response for chat
@app.route('/chat', methods=['POST'])
def chatRepsonse():
    global chatBot
    
    #Get the data from the request
    data = request.get_json()
    question = data['user']

    #Reset history / reload prompts
    if len(question) == 0:
        print("Reset LLM Chat state...")
        chatBot=ReloadLLM()
        return jsonify({'message' : "***RESET***"})
    
    #Ask question to fChain
    print("Running Question:" + question)
    result,debugmsgs=chatBot.chat(question)
    result=handleUrls(result)

    #show conversation size
    print("Tokens Used: "+str(chatBot.totalTokens))

    if chatBot.totalTokens>=  sizeBeforeCompact:
        print("Compacting LLM Chat state...")
        debugmsgs.append("Compacting LLM Chat state...")
        chatBot.Compact()

    #print("Answer:" + result)
    return jsonify({'message' : result,'debug':debugmsgs})

if __name__ == '__main__':
    app.run()

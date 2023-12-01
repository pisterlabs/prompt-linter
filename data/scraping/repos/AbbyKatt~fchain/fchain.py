#Prototype for fChain
#Simple OpenAI function based chatbot
import openai
import chromadb
import json
import os
from pathlib import Path
import errno

#---------------------------------------------------------------------------------
# cChain simple directory based logging
#---------------------------------------------------------------------------------
class fChainLog():
    def __init__(self,logFileDirectory):
        self.logFlogFileDirectory=logFileDirectory

        #Recursively try to make the directory if it doesn't exist
        try:
            os.makedirs(logFileDirectory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
        
        # #Create log file
        # self.logFile = open(logFileDirectory+"/log.txt", "w")
        # self.logFile.write("Log file created\n")
        # self.logFile.close()

    def Log(self,role,message,function_name,function_args):
        
        #Check self logging not none
        if self.logFlogFileDirectory is None:
            return

        #Make a unique filename with date and timestamp
        import datetime
        now = datetime.datetime.now()
        filename="Log_"+ role + "_" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + ".txt"

        #Create log file/write
        self.logFile = open(os.path.join(self.logFlogFileDirectory,filename), "a")
        self.logFile.write("role: "+role+"\n")
        self.logFile.write("function_name: "+function_name+"\n")
        self.logFile.write("function_args: "+str(function_args)+"\n")
        self.logFile.write("message: "+str(message)+"\n")
        self.logFile.close()

#---------------------------------------------------------------------------------
# fChain main class
#---------------------------------------------------------------------------------
class fChain():

    def __init__(self,SystemPrompt,functionList,debug=True,debugFunctions=False,debugFunctionStubs=False,logDir=None,nameChatBot="assistant",nameUser="user",model_name = "gpt-3.5-turbo"):
        self.functions={func.openai_schema["name"]: func for func in functionList}
        self.messages=[]
        self.nameChatBot=nameChatBot
        self.nameUser=nameUser
        self.model_name=model_name
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.debug=debug
        self.debugFunctions=debugFunctions
        self.debugFunctionStubs=debugFunctionStubs
        self.totalTokens=0

        #Setup logging
        self.Logging=fChainLog(logDir)

        #Load in SystemPrompt
        self.SystemPrompt=SystemPrompt
        self.addMessage(SystemPrompt,role="system")

    def addMessage(self,content,role="user",function_name=None):
        if function_name is not None:
            self.messages.append({"role": "function","name":function_name, "content": content})
            self.Logging.Log("function_call","",function_name,content)
        else:
            self.messages.append({"role": role, "content": content})
            self.Logging.Log(role,content,"","")

    def getMessages(self):
        return self.messages

    def getFunctionSchema(self):
        schemas=[]
        for aFunc in self.functions:
            schemas.append(self.functions[aFunc].openai_schema)
        return schemas
    
    def formatFunctionCall(self,function_name,arguments):
        argumentsString=""
        arguments=json.loads(arguments)
        for key in arguments:
            argumentsString+=str(key)+"="+str(arguments[key])+","
        argumentsString=argumentsString[:-1]
        #argumentsString = ','.join([key + '=' + str(arguments[key]) for key in arguments])
        return function_name + "(" + argumentsString + ")"

    def chat(self,userMessage,role="user"):
        #Add messge to list
        self.addMessage(userMessage,role=role)
        
        #get response passing in functions schemas
        response = openai.ChatCompletion.create(model=self.model_name,
                                        functions=self.getFunctionSchema(),
                                        temperature=0.2,
                                        messages=self.messages)
        if self.debug:
            print("------------------ GPT RESPONSE ------------------") 
            print(response)
            print("------------------ END RESPONSE ------------------\n")

        #Pevent infinite loop
        maxLoops=10
        currLoop=1
        self.totalTokens=response.usage.total_tokens

        #Loop until all functions have been called
        debugMsgs=[]
        finish_reason=response.choices[0].finish_reason
        while finish_reason=="function_call":

            #Get function name/run it/get response
            function_name=response.choices[0].message["function_call"]["name"]
            arguments=response.choices[0].message["function_call"]["arguments"]

            if self.debug or self.debugFunctionStubs:
                debugFuncMsg=self.formatFunctionCall(function_name,arguments)
                print("Running Function: ["+debugFuncMsg+"]")
                self.Logging.Log("function_call","",function_name,debugFuncMsg)
                if self.debugFunctionStubs:
                    debugMsgs.append(debugFuncMsg)

            function=self.functions[function_name]
            function_response=function.from_response(response)

            #Format json string nicely and human readable   
            if self.debugFunctions:
                print(json.dumps(json.loads(function_response), indent=4, sort_keys=True))
            if self.debug:            
                print("FINISHED: ["+function_name +"]")

            #Put response in messages queue
            self.addMessage(function_response,role="function",function_name=function_name)

            #Invoke GPT with message history, list of callable functions and schemas and the current message
            response = openai.ChatCompletion.create(model=self.model_name,
                                        functions=self.getFunctionSchema(),
                                        messages=self.messages)

            if self.debug:
                print("------------------ GPT RESPONSE ------------------") 
                print(response)
                print("------------------ END RESPONSE ------------------\n")
        
            if currLoop>maxLoops:
                print("Max loops reached!")
                break
            
            #Increment loop counter + get finish reason
            currLoop+=1
            finish_reason=response.choices[0].finish_reason

        #We're done - chuck the response in the messages queue
        messagetext=response.choices[0].message.content
        self.totalTokens=response.usage.total_tokens
        self.addMessage(messagetext,role="assistant")
        return (messagetext,debugMsgs)

    #Uses the AI to summarize the conversation then makes that the new message history reducing the token count
    def Compact(self):
        self.Logging.Log("compact","","","")        
        print("***Compacting chat history***")
        compactPrompt="Can you give me a brief summary of the conversation so in the third person narrative of both speakers?"
        ret=self.chat(compactPrompt,role="system")
        if self.debug:
            print("------------------ COMPACT SUMMARY ------------------") 
            print(ret)
            print("\n------------------ END SUMMARY ------------------\n")

        #Reset chat history
        self.messages=[]
        self.addMessage(self.SystemPrompt,role="system")
        self.addMessage("Please give me a summary of our current chat:")
        self.addMessage(ret,role="assistant")

        return ret
    

#---------------------------------------------------------------------------------
# JSON Loading of Knowledge Base as a semantic vectorstore
#---------------------------------------------------------------------------------
#Third version with configurable collection buckets
class fChainVectorDB():
    def __init__(
        self,
        file_path
        ):
        self.file_path = Path(file_path).resolve()
        
    #Reads a custom JSON and inserts it into multiple collections in chroma with metadata
    #returns chromaDB instance
    def load(self):
        vectorCollections={}
        # Load JSON file
        with open(self.file_path) as file:
            data = json.load(file)
            for item in data:
                collection=item['collection']
                metadata=item['metadata']
                page_content=item['chunk']

                #if collection is not in vectorCollections, add it
                if collection not in vectorCollections:
                    vectorCollections[collection]=[]
                #add the page content and metadata to the collection
                vectorCollections[collection].append((page_content,metadata))

        #Setup local client        
        client = chromadb.Client()
        
        for colName in vectorCollections:
            collection = client.create_collection(colName)
            #take each tuple from list data and turn it into new lists docs and metas
            docs = [x[0] for x in vectorCollections[colName]]
            metas = [x[1] for x in vectorCollections[colName]]
            idx = ["Doc_{0}".format(i) for i in range(len(vectorCollections[colName]))]

            # Add docs to the collection. Can also update and delete. Row-based API coming soon!
            collection.add(
                documents=docs,
                metadatas=metas,
                ids=idx, # unique for each doc
            )

        return client        


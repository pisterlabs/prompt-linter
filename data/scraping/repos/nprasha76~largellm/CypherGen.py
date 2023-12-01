from langchain import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import ChatPromptTemplate

from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
import os





class CypherGen:

 def __init__(self):

   print ('In self')
   if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)



   model_n_ctx = os.environ.get("MODEL_N_CTX")

   callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

   self.llm = LlamaCpp(
    model_path="./models/zephyr-7b-beta.Q4_K_M.gguf",
    #model_type="mistral",
    n_ctx=8192, #5000
    n_gpu_layers=1,
    n_batch=1, #512
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    temperature=0.0,
    streaming=False,
    verbose=True,
    )
   

    # Prompt
   context ="Generate cypherQL statement in Neo4J for the given question"


   self.egtemplate1 = """<s>[INST]You are a helpful cypherQL code assistant. Based on following example "key":"value" pairs ,
                    Your task is to strictly deriver  cypherQL statement only without explanations.

                    -"Get CMD jobs that are part of BOX 'B' ":"MATCH(c:CMD)<-[CONTAINS]-(d:BOX) where d.insert_job='B' return c,d ;"
                    
                    -"Get CMD jobs that belongs to  BOX 'B' ":"MATCH(c:CMD)<-[CONTAINS]-(d:BOX) where d.insert_job='B' return c,d ;"
                    
                    -"Get jobs of BOX 'B' ":"MATCH()<-[CONTAINS]-(d:BOX) where d.insert_job='B' return d ;"
                    
                    -"Get jobs that belong to  BOX 'B' ":"MATCH()<-[CONTAINS]-(d:BOX) where d.insert_job='B' return d ;"
                    
                    -"Get jobs that are part of BOX 'B' ":"MATCH()<-[CONTAINS]-(d:BOX) where d.insert_job='B' return d ;"
                    
                    -"Get CMD jobs that triggers CMD 'CMD_JOB'" :"MATCH(c:CMD)-[TRIGGERS]->(d:CMD)  where d.insert_job='CMD_JOB' return c,d ;"
                    
                    -"Get CMD jobs that invokes CMD 'CMD_JOB'" :"MATCH(c:CMD)-[TRIGGERS]->(d:CMD)  where d.insert_job='CMD_JOB' return c,d ;"
                    
                    -"Get CMD jobs that are output to SYSTEM 'OPS'":"MATCH(c:CMD)-[OUTPUT]->(d:SYSTEM) where d.insert_job='OPS' return c,d ;"
                    
                    -"Get CMD jobs that have input from  SYSTEM 'OPS'":"MATCH(c:CMD)<-[INPUT]-(d:SYSTEM) where d.insert_job='OPS'  return c,d ;"
                    
                    -"Get CMD jobs that are triggered by CMD 'CMD_JOB'" :"MATCH(d:CMD)<-[TRIGGERS]-(c:CMD)  where c.insert_job='CMD_JOB' return c,d ;"
                    
                    -"Get jobs that are output to SYSTEM 'OPS'":"MATCH()-[OUTPUT]->(d:SYSTEM) where d.insert_job='OPS' return d ;"
                    
                    -"Get jobs that have input from  SYSTEM 'OPS'":"MATCH()<-[INPUT]-(d:SYSTEM) where d.insert_job='OPS'  return d ;"
                    
                    -"Get jobs of SYSTEM 'OPS'":"MATCH()-[]-(d:SYSTEM) where d.insert_job='OPS' return d ;"
                    
                    -"Get jobs that belong to SYSTEM 'OPS'":"MATCH()-[]-(d:SYSTEM) where d.insert_job='OPS' return d ;"

                    -"Get CMD jobs that triggers FW 'FW_JOB'" :"MATCH(c:CMD)-[TRIGGERS]->(d:FW)  where d.insert_job='FW_JOB' return c,d ;"
                    
                    -"Get CMD jobs that are triggered by FW 'FW_JOB'" :"MATCH(d:CMD)<-[TRIGGERS]-(c:CMD)  where c.insert_job='FW_JOB' return c,d ;"
                    
                    -"Get FileWatcher jobs that are triggered by CMD 'CMD_JOB'" :"MATCH(d:CMD)-[TRIGGERS]->(c:FW)  where d.insert_job='CMD_JOB' return c,d ;"
                    
                    -"Get FileWatcher jobs that triggers CMD 'CMD_JOB'" :"MATCH(c:FW)-[TRIGGERS]->(d:CMD)  where d.insert_job='CMD_JOB' return c,d ;"
                    
                    [/INST]
                    
                    </s>
                    [INST]
                        Strictly deriver cypherQL statement only without explanations for  : "{question}":
                    [/INST]

                    """




   egtemplate2 = """<s>[INST]You are a helpful cypherQL code assistant. Based on following sample question and answer pairs given ,
                    your task is to deriver cypherQL statment for the "new question" without explanations.

                    "question":"Get CMD jobs that are part of BOX B "
                    "answer":"MATCH(c:CMD)<-[CONTAINS]-(b:BOX) where b.insert_job='B' return c,b ;"
                    

                    "question":"Get CMD jobs that triggers CMD_JOB" 
                    "answer":"MATCH(c:CMD)-[TRIGGERS]->(d:CMD)  where d.insert_job='CMD_JOB' return c,d ;"
                    
                    "question":"Get CMD jobs that are output to SYSTEM OPS"
                    "answer":"MATCH(c:CMD)-[OUTPUT]->(d:SYSTEM) where d.insert_job='OPS' return c,d ;"
                    
                    "question": "Get CMD jobs that have input from  SYSTEM OPS"
                    "answer":"MATCH(c:CMD)<-[INPUT]-(d:SYSTEM) where d.insert_job='OPS'  return c,d ;"
                
                    [/INST]
                    Just derive cypherQL statement for below "new question" without explanations.
                    "new question":{question}
                    
                    </s>

                    """


   prompt = """
    Question: {question}
    """
   promptTemplate = PromptTemplate(template=prompt, input_variables=["question"])
    

    #question = "How can I initialize a ReAct agent?"


    # Run
    #print ("Running chain with question " ,question)
    #llm(question)
    


# this works

 def invokeLLM(self):
    user_prefix = "[user]: "
 
    while True:
            question = input(user_prefix)
            question =question.lower().replace("invokes","triggers").replace("calls","triggers")
            question =question.lower().replace("invoke","triggers").replace("calls","triggers")
            question =question.lower().replace("invoked","triggered").replace("called","triggered")
            question =question.lower().replace("holds","contains").replace("consists","contains")
            question =question.lower().replace("hold","contains").replace("consists","contains")
            #generator = llm(prompt)
            
            #context ="Generate cypherQL statement in Neo4J for the given question"
            promptTemplate = PromptTemplate(template=self.egtemplate1, input_variables=["question"])

            llm_chain = LLMChain(prompt=promptTemplate, llm=self.llm)
            response=llm_chain.run(question)
            
            print (response)
            print("")



obj=CypherGen()
obj.invokeLLM()
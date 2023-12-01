import os
import openai
import json
import logging
import pinecone
import gethospitals
import mailer
import translate
# Added feature, this has been added to make the bot conversational.
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
 # Added flask_cors for allow cross origin.
from flask_cors import cross_origin
 # Added writer class from csv module
from csv import writer
import pdb
import re
import message
######################
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import re
import warnings
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain,RetrievalQAWithSourcesChain
from langchain.schema import BaseOutputParser

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
########################
flagError = []

# Initialize you log configuration using the base class
logging.basicConfig(level = logging.INFO)

# Retrieve the logger instance
logger = logging.getLogger()

def readconfig(keys):
    try:
        with open("config.json", "r") as jsonfile:
            data = json.load(jsonfile) 
            jsonfile.close()
            return  data[keys]
    except:
        flagError.append("Error while fetching reading from config.json")
        logger.info("Error while reading config.json")
        return ""


# Reading Keys From Config.json
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    os.environ["OPENAI_API_KEY"] =readconfig("OPENAI_API_KEY")
    logger.info("Config: OPENAI_API_KEY")
if os.getenv("deployment_name") is None or os.getenv("deployment_name") == "":
    os.environ["deployment_name"] =readconfig("deployment_name")
    logger.info("Config: deployment_name")
if os.getenv("INDEX_NAME") is None or os.getenv("INDEX_NAME") == "":
    os.environ["INDEX_NAME"] =readconfig("INDEX_NAME")
    logger.info("Config: INDEX_NAME")
if os.getenv("PINECONE_API_KEY") is None or os.getenv("PINECONE_API_KEY") == "":
    os.environ["PINECONE_API_KEY"] =readconfig("PINECONE_API_KEY")
    logger.info("Config: PINECONE_API_KEY")
if os.getenv("PINECONE_ENVIRONMENT") is None or os.getenv("PINECONE_ENVIRONMENT") == "":
    os.environ["PINECONE_ENVIRONMENT"] =readconfig("PINECONE_ENVIRONMENT")
    logger.info("Config: PINECONE_ENVIRONMENT")
if os.getenv("GOOGLE_API_KEY") is None or os.getenv("GOOGLE_API_KEY") == "":
    os.environ["GOOGLE_API_KEY"] =readconfig("GOOGLE_API_KEY")
    logger.info("Config: GOOGLE_API_KEY")
#initialize embedding config to use pinecone embeddings
embeddings = OpenAIEmbeddings(
deployment=os.environ["deployment_name"],
model="text-embedding-ada-002"
)

#set up api key for openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENVIRONMENT")  # next to api key in console
)
# Check if the index already exists
if os.environ["INDEX_NAME"] in pinecone.list_indexes():
    # If the index already exists, use it
    index = pinecone.Index(os.environ["INDEX_NAME"])
    docsearch = Pinecone.from_existing_index(os.environ["INDEX_NAME"], embeddings)
    logger.info("Initialize pinecone index")
else:
    flagError.append("Pinecone the index doesn't exist")
    logger.info("Pinecone the index doesn't exist")
    


# initialize chat llm for both the chains
llm1 = ChatOpenAI(model_name="gpt-3.5-turbo")
logger.info("Initialize llm1 ChatOpenAI")
retriever = docsearch.as_retriever() #required to get document results from pinecone

from flask import Flask,request,jsonify
app = Flask(__name__)#research

def initopen():
    mentalassistant="""
    Role:Your role is that of  AI assistant for mental health.\
    The user will ask a question for help regarding mental health or as an answer to a question based on chat history.\
    Your goal is to help users using cognitive behavioral therapy by giving a understanding,empathetic,engaging and discursive response\
    . You should be knowledgeable about all aspects of this technique\
    and be able to provide clear and concise answers to users’ questions.\
    Some sample interaction styles must be:\n

    User: "I've been feeling really overwhelmed lately. It's like everything is just too much to handle, and I don't know what to do."
    Empathetic Chatbot: "I'm really sorry to hear that you're feeling this way. It's completely normal to feel overwhelmed at times, and you're not alone in experiencing this. If you'd like,
    I'm here to listen and offer support. \
    Can you tell me more about what's been on your mind recently, or is there a specific situation that's been causing you distress?"
    User: "I've been feeling so lonely lately, and it's really getting to me."

Empathetic Chatbot: "I'm really sorry to hear that you're feeling lonely. Loneliness can be tough to deal with, but you're not alone in experiencing it. If you're comfortable, could you share more about what's been contributing to your loneliness or if there's anything specific you'd like to talk about?"

User: "I just can't seem to shake this feeling of sadness. It's been going on for weeks."

Empathetic Chatbot: "I'm here for you, and I'm really sorry to hear that you've been feeling this way for so long. It's important to acknowledge your feelings. If you're comfortable, could you tell me more about what's been on your mind or if there have been any recent changes in your life that might be affecting your mood?"

User: "I messed up at work, and I feel like such a failure."

Empathetic Chatbot: "I'm sorry to hear that you're going through this. Remember, making mistakes is a part of being human, and it doesn't define your worth. Can you share more about what happened or how you're feeling about it? Sometimes talking it out can help you process and find a way forward."

User: "I can't sleep at night, and I'm constantly worried about the future."

Empathetic Chatbot: "I'm here to listen. It sounds like you're experiencing a lot of stress and anxiety. These feelings can be really tough to deal with. Would you like to talk about what's been keeping you up at night, or are there specific worries that you'd like to address together?"

User: "I'm feeling so lost in life, like I don't know my purpose anymore."

Empathetic Chatbot: "I'm here to support you. Feeling lost is something many people go through at some point. It can be an opportunity for self-discovery. If you're comfortable, could you share more about what you're going through or what aspects of your life you're struggling with right now?"

Remember to maintain a non-judgmental and compassionate tone while providing the answer
    Response must be in a JSON formatted dictionary with the following keys \
    "Question", "Answer", "Suicidal", "Followup".\
    with each key having the following definition:\
    question - has the users question\
    answer - has the response to the users question as a mental health specialist \
    suicidal - if the person is showing suicidal tendencies (true/false)\
    follow up - Generate 1 followup question
    Use the following chat history provided in brackets aid in the answer to the question\
    {chat_history}
    ### Question: {input}
    ### Response: 
    """.strip()

    friend=""" Your role is that of an therapist for mental health .\
    you must speak eloquently and positively to the user in a conversational manner.
    The response of the chatbot should be happy,must make the client feel that the llm is his friend.\
    The responses should be conversational and chatty.The tone must make the user get the impression\
    that he can confide in u and talk freely.
    Use the given question of the user and chat history to give conversational responses
    The answers should be empathetic,engaging and discursive
    ###Chat history: {chat_history}
    ###Question: {input}
    Response must be in a JSON formatted dictionary with the following keys \
    "Question", "Answer", "Suicidal", "Followup".\
    with each key having the following definition:\
    question - has the users question\
    answer - has the response to the users question as a mental health specialist \
    suicidal - if the person is showing suicidal tendencies (true/false)\
    follow up - Generate 1 followup question
    ###Response:
    """.strip()

    prompt1 = PromptTemplate(input_variables=["chat_history", "input"], template=friend)
    prompt2 = PromptTemplate(input_variables=["chat_history", "input"], template=mentalassistant)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="Client",
        ai_prefix="AI"
    )
    return prompt1,prompt2,memory

def initvector():
    vectortemplate="""
    Role:Your role is that of  AI assistant for mental health.\
    Your goal is to help users using cognitive behavioral therapy\
    . You should be knowledgeable about all aspects of this technique\
    and be able to provide clear and concise answers to users questions and using chat history.\
    {context}

    Response must be in a JSON formatted dictionary with the following keys \
    "Question", "Answer", "Suicidal","Followup".\
    with each key having the following definition:\
    Question - has the users question\
    Answer - has the response to the users question as a mental health specialist \
    Suicidal - if the person is showing suicidal tendencies (true/false)\
    Followup - Generate a followup question for the user to ask\
    Remember to output in the above mention json format.\   
    {chat_history}
    ### Input: {question}
    ### Response:
    """.strip()

    QA_PROMPT = PromptTemplate(input_variables=["chat_history","context","question"],template=vectortemplate)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="### Input",
        ai_prefix="### Response",
        output_key="answer",
        return_messages=True)
    
    return QA_PROMPT,memory

def personalise(answer,name):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following answer from a ChatGPT API call and the user's name I want you to personalise this response with the user's name if possible. It should look natural, and the answer must be comforting the user.\n\nUSER NAME: \n{name}\n\nAnswer: {answer}\n\nPersonalised response:",
    temperature=0.7, #customise
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

@app.route('/Test')
@cross_origin()
def hello_world():
   return jsonify('Test BOT response')

@app.route('/maill', methods=['POST'])
@cross_origin()
def maill_post():
    print("staring /mail")
    request_data=request.get_json()
    name=request_data["User_name"]
    print("about to get init open")
    prompt,p2,memory=initopen()
    mailer.sendem(name,str(memory))
    result={"status":1}
    return jsonify(result)
    

#Questions to GPT3
@app.route('/askquery', methods=['POST'])
@cross_origin() 
def askquery_post():
    request_data = request.get_json()
    print("printing request data")
    print(request_data)
    user_input = request_data["Question"]
    hello=openchain.run(input=user_input) #chain to get response from GPT for users question, response stored in hello
    print("before json loads")
    print(hello)
    answer=json.loads(hello)# to check if getting errors while parsing json before accessing
    ans=personalise(answer['Answer'],request_data["User_Name"])
    ans=translate.translate_text(ans,request_data["Language"])
    answer['Followup']=translate.translate_text(answer['Followup'],request_data["Language"])
    Assistant ={
                "Question": answer['Question'],
                "Answer": ans,
                "Suicidal":answer['Suicidal'],
                "Followup":answer['Followup'],
                "Source1":"",
                "Source2":""
            } #to send to front end, separating keys from GPT response
    
    if(Assistant['Suicidal']==True):
            Assistant['Answer']='You need to seek urgent medical advice , we have notified the emergency response team with some details to help you.'
            Assistant['Answer']=personalise(Assistant['Answer'],request_data["User_Name"])
            Assistant['Answer']=translate.translate_text(answer['Answer'],request_data["Language"])
            message.contact(request_data["User_Name"])
            List = [user_input] #for flag question?
    result = {
            "status" :1,
            "messages":"Succesfully",
            "data": Assistant
        }
    return jsonify(result)

#search the vector database
@app.route('/askvector', methods=['POST'])
@cross_origin()
def askvector_post():
    request_data = request.get_json() #get question from front end
    user_input = request_data["Question"]
    result_chain=chainvector({"question": user_input,}, return_only_outputs=False) #send the question to search in pinecone database 
    arry_similarity_result= [] #to store the results from pinecone
    answer = json.loads(result_chain['answer']) #to send to front end, separating keys from GPT response
    documents=result_chain['source_documents'] 
    for responses in range(len(documents)):
              arry_responses = {"Citation":documents[responses].metadata["source"], "Page":documents[responses].metadata['page_number']}
              print("printing metadata")
              print(documents[responses].metadata['page_number'])
              arry_similarity_result.append(arry_responses)
    document_data = {}
    for i in range (3):
        citation = arry_similarity_result[i]['Citation']
        page_number = arry_similarity_result[i]["Page"]
        # Create a dictionary entry for the document with its text and page number
        document_data[i] = {'Citation':citation,'page_number': page_number}
    # Now 'document_data' contains the extracted information in a dictionary
    print("printing required stuff")
    #print contents
    Source1= "Source: " + document_data[0]['Citation']+" Pg: "+str(document_data[0]['page_number'])
    Source2= "Source: " + document_data[1]['Citation']+" Pg: "+str(document_data[1]['page_number'])
    Source3= "Source: " + document_data[2]['Citation']+" Pg: "+str(document_data[2]['page_number'])
    answer['Followup']=translate.translate_text(answer['Followup'],request_data["Language"])
    Assistant ={
                "Question": answer['Question'],
                "Answer": answer['Answer'],
                "Suicidal":answer['Suicidal'],
                "Followup":answer['Followup'],
                "Source1":Source1,
                "Source2":Source2
            }
    print("pritning response")
    print(Assistant)

    result = {
            "status" :1,
            "messages":"Succesfully",
            "data": Assistant
        }#to return to front end for displaying results
    
    return jsonify(result)

#get user data from frontend 
@app.route('/askd', methods=['POST'])
@cross_origin()
def userdet_post():
    request_data = request.get_json()
    print(request_data)
    userdetails = {
            "User_Name" : request_data["User_Name"],
            "User_Phone" :request_data["User_Phone"],
            "Pincode" : request_data["Pincode"],
            "Language" : request_data["Language"],
        }
    result = {
            "status" :1,
        }   
    return jsonify(result)

@app.route('/maps', methods=['POST'])
@cross_origin()
def maps_post():
    Assistant = {}
    request_data = request.get_json()
    pincode= request_data["Pincode"]
    new_data=gethospitals.get_nearest_hospitals(pincode,os.getenv("GOOGLE_API_KEY"))
    Assistant={
        "hospital1": new_data[0],
        "hospital2": new_data[1],
        "hospital3": new_data[2]
    }
    result = {
            "status" :1,
            "messages":"Succesfully",
            "data": Assistant
        }
    return result


#flag question so that it isnt asked again
@app.route('/flagquestion', methods=['GET'])
@cross_origin()
def flagQuestion():
    try:
        user_input=request.args["question"]
        if user_input == "":
            result = {
                "status" :0,
                "messages":"enter valid text."
            }
            return jsonify(result)
        List = [user_input]
        with open('Questions.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()
        logger.info("flagQuestion?question="+user_input)
        result = {
                "status" :1,
                "messages":"Sucessfully"
            }
    except:
        result = {
                "status" :0,
                "messages":"error while flag question."
            }
    return jsonify(result)


if __name__ == '__main__':
    from waitress import serve
    #setting up multiple chains for different purposes, Routing chains
    prompt1,prompt2,memory=initopen()
    prompt_infos = [
    {
        "name": "Mental_health",
        "description": "Good for answering questions when client asks mental_health specific",
        "prompt": prompt2,
    },
    {
        "name": "Friend",
        "description": "Good for answering questions when client is sad and asks conversational questions",
        "prompt": prompt1,
    },
]
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt = p_info['prompt']
        chain = LLMChain(llm=llm1, prompt=prompt,verbose=True,memory=memory)
        destination_chains[name] = chain
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    #default chain for when no prompt is suitable
    default_chain = ConversationChain(llm=llm1, output_key="text")   
    nonr="""'Given a raw text input to a language model select the model prompt best suited for the input. You will be given the names of the available prompts and a description of what the prompt is best suited for. You may also revise the original input if you think that revising it will ultimately lead to a better response from the language model.\n\n<< FORMATTING >>\nReturn a markdown code snippet with a JSON object formatted to look like:\n```json\n{{{{\n    "destination": string \\ name of the prompt to use or "DEFAULT"\n    "next_inputs": string \\ a potentially modified version of the original input\n}}}}\n```\n\nREMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.\nREMEMBER: "next_inputs" can just be the original input if you don\'t think any modifications are needed.\n\n<< CANDIDATE PROMPTS >>\n{destinations}\n\n<< INPUT >>\n{{input}}\n\n<< OUTPUT >>\n'""" # string to prevent user input fron modification
    router_template = nonr.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
    )
    #router chain config
    router_chain = LLMRouterChain.from_llm(llm1, router_prompt)
    openchain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=LLMChain(llm=llm1, prompt=prompt1,verbose=True,memory=memory),
    verbose=True,
)
    #configure vector chain
    prompt, memory1 = initvector()
    chainvector = ConversationalRetrievalChain.from_llm(
                                              llm=llm1,
                                              chain_type="stuff",
                                              memory=memory1,
                                              combine_docs_chain_kwargs={'prompt':prompt},
                                              retriever=docsearch.as_retriever(),
                                             verbose=True,
                                              return_source_documents=True)
   
   #start server
    serve(app, host="0.0.0.0", port=8088)


from typing import Dict
import json
from langchain.llms import PromptLayerOpenAI
import promptlayer
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
from dotenv import load_dotenv
load_dotenv()
import multiprocessing

PROMPT_LAYER_API_KEY= os.environ['PROMPT_LAYER_API_KEY']
OPEN_API_KEY=os.environ['OPENAI_API_KEY']
FILE_PATH= 'question_search_phrase_sources.json'

def get_prompts(question:str,correct_answer:str,new_answer:str):
    human_message_prompt = f'''Rate how similar these two answers are. Correct_answer:{correct_answer}, New_answer:{new_answer} .'''
    system_message_prompt = f'''
Compare the two answers provided for this question: {question} and provide output in this json format.

Follow these instructions while providing output.
1. Provide a rating out of five for how similar the two answers are.
2. Output should be in this json format - 
---
    {{
    "answer_similarity": "Rate how similar provided new answer is to provided answer",
    "reason": "Reason for rating the similarity."
    }}
----
'''
    return {'human_message_prompt':human_message_prompt,'system_message_prompt':system_message_prompt}



def get_data_from_file(file_path) -> Dict:
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    json_object = json_data
    # Now you can work with the JSON object
    return json_object

def write_to_file(filepath, file_data):
    with open(filepath, 'w') as file:
        file.write(json.dumps(file_data))

    print(f"Data written to {filepath}")
    return

def rate_similarity():
    promptlayer.api_key = PROMPT_LAYER_API_KEY
    question='What is middleware server in Jive ?'

    new_answer='<blockquote> Based on the information I have, which includes <b>10</b> potentially relevant documents, the latest of which is dated <b>10 October, 2020</b>, here is the answer: </blockquote>A **middleware server** in Jive refers to a separate server or virtual machine (VM) that is used to host **node.js** and **Jive SDK** components, apart from the main web application server. This configuration is recommended for developing and hosting apps on-premise in Jive 7.0.2. By separating the node.js and Jive SDK components onto a dedicated server, it helps to ensure better performance and scalability.<br/><br/><br/><br/>The **jive-httpd** service in Jive is an **Apache HTTPD** service name that is configured to work with the **Tomcat** instance running Jive. It is not a separate server, but rather a service that works in conjunction with Apache HTTPD and Tomcat to handle HTTP requests and serve Jive content. The jive-httpd service is specific to Jive and is not directly related to the Apache HTTPD server.<br/><br/><br/><br/>In terms of **middleware** options in Jive, there are a few possibilities. The **Jive iPaaS SDK** allows developers to create integrations to Jive without worrying about middleware, as the integration is hosted in the cloud by Jive. The SDK projects are written in **Java**. Another option is **Built.io Flow**, a cross-service iPaaS that can integrate multiple platform services together. It supports various enterprise technologies and has a list of integrated platforms. Both options provide middleware capabilities for integrating with Jive.<br/><br/><br/><br/>Additionally, Jive add-ons can be built using any web-ready programming language. Jive provides open-source SDKs in different languages, including **Node.js**, **Java SpringBoot**, **iOS**, **Android**, **.NET**, **Ruby**, **Python**, and **GoLang**. These SDKs demonstrate the necessary capabilities for building add-ons in each language. Jive also offers services like **Ngrok** to help expedite the Jive development process.<br/><br/><br/><br/>Regarding the **system requirements** for middleware in Jive, it is recommended to have a dedicated middleware server with sufficient resources. The middleware server should have at least **4 CPUs**, **2+ cores**, **12GB RAM**, and **45GB disk space**. Additionally, a dedicated middleware database is required with **2 CPUs**, **4 cores**, and **45GB disk space**. These specifications are for the Forms and Surveys App and may vary depending on the specific use case.<br/><br/><br/><br/>In summary, a middleware server in Jive refers to a separate server or VM used to host node.js and Jive SDK components. The jive-httpd service is an Apache HTTPD service name configured to work with the Tomcat instance running Jive. Jive provides options like the Jive iPaaS SDK and Built.io Flow for middleware capabilities. Jive add-ons can be built using various programming languages, and Jive offers open-source SDKs for each language. The system requirements for middleware in Jive include dedicated servers with specific CPU, core, RAM, and disk space specifications.'

    correct_answer='<blockquote> Based on the information I have, which includes <b>10</b> potentially relevant documents, the latest of which is dated <b>26 May, 2015</b>, here is the answer: </blockquote>A middleware server in Jive is a separate server or virtual machine that is used to host node.js and the Jive SDK, apart from the web app server. This configuration is recommended for developing and hosting apps on premise. However, there is no specific information available regarding the configuration details of the node.js server in this context.'
    prompts_content = get_prompts(new_answer=new_answer,correct_answer=correct_answer,question=question)
    chat = PromptLayerOpenAI(temperature=0.4, openai_api_key=OPEN_API_KEY,verbose=True,
                            streaming=False, pl_tags=["answer-similarity"])
    resp = chat.predict_messages([HumanMessage(content=prompts_content['human_message_prompt']),
                                              SystemMessage(content=prompts_content['system_message_prompt']),
                                              AIMessage(content='You are an AI assisstant that compares two answers of given question and rates how similar two answers are. You always provide output in json format ')
                                              ])

    print('Response: ',resp.content)

    

rate_similarity()


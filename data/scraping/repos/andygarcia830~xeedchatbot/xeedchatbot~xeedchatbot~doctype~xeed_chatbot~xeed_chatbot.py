# Copyright (c) 2023, Xurpas Inc. and contributors
# For license information, please see license.txt

import frappe,os,json,openai
from langchain.llms import OpenAI
#from llama_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from llama_index import StorageContext, load_index_from_storage,LLMPredictor,ServiceContext, PromptHelper
from frappe.model.document import Document
from langchain.callbacks.base import BaseCallbackHandler

class XeedChatbot(Document):
	pass

class XeedResponseHandler(BaseCallbackHandler):
    
    def __init__(self,session):
         self.session=session

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")
        frappe.publish_realtime(self.session,token)


@frappe.whitelist(allow_guest=True)
def get_sid():
     return frappe.request.cookies.get('sid','Guest')

@frappe.whitelist(allow_guest=True)
def login():
    user='ailab@xurpas.com'
    pwd='ailab2023'
    try:
          manager=frappe.auth.LoginManager()
          manager.authenticate(user=user,pwd=pwd)
          manager.post_login()
    except frappe.exceptions.AuthenticationError:
        frappe.clear_messages() 
        frappe.local.response['message']={
             'success_key':0,
             'message':'Authentication Error'
        }
        return
    return frappe.request.cookies.get('sid',frappe.session.user)

    

@frappe.whitelist(allow_guest=True)
def config():
     configdoc= frappe.get_doc('Xeed Chatbot Frontend Settings')
     response={}
     response['headerImgUrl']=configdoc.header_image
     response['headerBgColor']=configdoc.header_background_color
     response['userAvartarBgColor']=configdoc.user_avatar_background_color
     response['businessAvatarBgColor']=configdoc.business_avatar_background_color
     response['businessAvatarImgUrl']=configdoc.business_avatar_image
     response['borderColor']=configdoc.border_color
     response['footerImgUrl']=configdoc.footer_image
     response['footerLink']=configdoc.footer_link
     response['footerText']=configdoc.footer_text
     response['token']=configdoc.token
     response['socketUrl']=configdoc.socket_url
     response['favIcon']=configdoc.favorite_icon
     
     return response

def api_key():
    os.environ["OPENAI_API_KEY"] = frappe.get_doc('Xeed Chatbot OpenAI Settings').openai_api_key
    openai.api_key= os.environ["OPENAI_API_KEY"]
    # key=os.environ["OPENAI_API_KEY"]
    # print(f'OPENAI KEY={key}')


@frappe.whitelist()
def ask_question(msg,jsonStr,gpt_session='default'):
    print(f'GPT SESSION={gpt_session}')
    print(f'JSON={jsonStr}')
    settings = frappe.get_doc('Xeed Chatbot GPT Settings')
    prompt = settings.guardrails
    index_folder=settings.index_folder
    if prompt == None:
         prompt = ""
    prompt=prompt +"""
    The Question is:
    """
    prompt = prompt+msg 
    print(prompt)
    jsonDict=json.loads(jsonStr)
    api_key()
    storage_context = StorageContext.from_defaults(persist_dir=index_folder)
    #index = GPTVectorStoreIndex.load_from_disk('/home/andy/frappe-bench/apps/aimgpt/aimgpt/private/')
    index = load_index_from_storage(storage_context)

    # GPT-4 Service Context

    llm = 'gpt-4'
    gpt_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name=llm,streaming=True, callbacks=[XeedResponseHandler(gpt_session)]))

    # define prompt helper
    # set maximum input size
    max_input_size = 8191
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 0.5
    gpt_prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=gpt_predictor, prompt_helper=gpt_prompt_helper)

    response = index.as_query_engine(service_context=service_context).query(prompt)
    jsonDict.append((msg.replace('"','\''),response.response.strip('\n').replace('"','\'')))
    return jsonDict



@frappe.whitelist()
def fetch_sample_questions():
    settings = frappe.get_doc('Xeed Chatbot GPT Settings')
    questions = settings.sample_questions
    print(f'QUESTIONS={questions}')
    result=[]
    for item in questions:
          result.append(item.question)
    ip = frappe.get_request_header('X-Forwarded-For')
    print(f'REQUEST={ip}')
    return result
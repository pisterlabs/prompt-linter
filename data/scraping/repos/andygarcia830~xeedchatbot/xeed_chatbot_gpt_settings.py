# Copyright (c) 2023, Xurpas AI Lab and contributors
# For license information, please see license.txt

import frappe,openai
from frappe.utils import get_site_name
from frappe.model.document import Document
from langchain.llms import OpenAI
import os
from llama_index import StorageContext, load_index_from_storage

from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
import sys


class XeedChatbotGPTSettings(Document):
	pass

def api_key():
    os.environ["OPENAI_API_KEY"] = frappe.get_doc('Xeed Chatbot OpenAI Settings').openai_api_key
    openai.api_key= os.environ["OPENAI_API_KEY"]
    # key=os.environ["OPENAI_API_KEY"]
    # print(f'OPENAI KEY={key}')


@frappe.whitelist()
def train():
	api_key()
	maindoc = frappe.get_doc('Xeed Chatbot GPT Settings')
	data_folder=get_site_name(frappe.local.request.host)+'/private/files/xeedchatbot/data'
	index_folder=get_site_name(frappe.local.request.host)+'/private/files/xeedchatbot/index'
	maindoc.data_folder=data_folder
	maindoc.index_folder=index_folder
	maindoc.save()
	traintext = ''
	for item in maindoc.chatbot_data:
		print(f'DATA={item}')
		
		text = ''
		if not item.text == None:
			text = item.text

		traintext=traintext+'\n\n'+text
	try:
		os.makedirs(data_folder)
	except:
		pass
	f = open(data_folder+'/training_data.txt', "w")
	f.write(traintext)
	f.close()
	construct_index(data_folder,index_folder)
	frappe.msgprint('Model Training Complete')



def construct_index(data_folder,index_folder):
    max_input_size = 4096
    num_outputs = 256
    max_chunk_overlap = .9
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-embedding-ada-002", max_tokens=num_outputs))
    print('LOADING DOCS FROM DIRECTORY')
    documents = SimpleDirectoryReader(data_folder).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    print('INDEXING')
    index =  GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    print('SAVING INDEX TO DISK')
    index.storage_context.persist(index_folder)
   
    return index



# Copyright (c) 2023, Xurpas Inc. and contributors
# For license information, please see license.txt

import frappe,os,openai,re
from frappe.model.document import Document
from frappe.utils import get_site_name
from langchain.llms import OpenAI
import os
from llama_index import StorageContext, load_index_from_storage

from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext

class World(Document):
	pass


def api_key():
    os.environ["OPENAI_API_KEY"] = frappe.get_doc('AI NPC OpenAI Settings').openai_api_key
    key=os.environ["OPENAI_API_KEY"]
    openai.api_key=key
    

@frappe.whitelist()
def train(world):
    maindoc = frappe.get_doc('World',world)
    print(f'WORLD={world}')
    for item in maindoc.npcs:
        print(f'NPC={item}')
        worldname = re.sub('[^0-9a-zA-Z]+', '_',maindoc.name)
        npc = re.sub('[^0-9a-zA-Z]+', '_',item.name)
        directory=get_site_name(frappe.local.request.host)+'/private/files/'+worldname+'/'+npc
        item.data_path=directory
        traindir = directory+'/train'
        story = ''
        if not item.story == None:
            story = item.story

        traintext=maindoc.lore+'\n\n'+story
        try:
            os.makedirs(traindir)
        except:
            pass
        f = open(traindir+'/backstory.txt', "w")
        f.write(traintext)
        f.close()
        construct_index(traindir,directory)

    maindoc.save()


def construct_index(train_path,save_path):
    api_key()
    max_input_size = 4096
    num_outputs = 256
    max_chunk_overlap = .9
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-embedding-ada-002", max_tokens=num_outputs))
    print('LOADING DOCS FROM DIRECTORY')
    documents = SimpleDirectoryReader(train_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    print('INDEXING')
    index =  GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    print('SAVING INDEX TO DISK')
    index.storage_context.persist(save_path)
   
    return index


@frappe.whitelist()
def delete_npc(world,name):
    npclist = frappe.db.get_list('NPC Interaction',filters={
        'npc': name
    })
    
    for item in npclist:
        print(f'DELETING {item}')
        frappe.db.delete('NPC Interaction',{'npc':name})
        frappe.db.commit()
    


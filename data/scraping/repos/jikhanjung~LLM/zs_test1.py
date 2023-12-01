from openai import OpenAI
from dotenv import load_dotenv # pip install python-dotenv
from pyzotero import zotero
import cmd
import os
import streamlit as st
from ZWrapper import ZWrapper

import requests

# disable ssl warning
requests.packages.urllib3.disable_warnings()

# override the methods which you use
requests.post = lambda url, **kwargs: requests.request(
    method="POST", url=url, verify=False, **kwargs
)

requests.get = lambda url, **kwargs: requests.request(
    method="GET", url=url, verify=False, **kwargs
)

load_dotenv()
        
client = OpenAI()
def get_or_create_assistant( asst_name ):
    asst_list = client.beta.assistants.list( order="desc", limit="20", )
    #print(asst_list.data)

    if len(asst_list.data) == 0:
        print("no assistant")
        assistant = client.beta.assistants.create(
            name=asst_name,
            instructions="You are a research assistant in paleontology.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4-1106-preview"
        )
        asst_list = client.beta.assistants.list( order="desc", limit="20", )

    for asst in asst_list.data:
        if asst.name == asst_name:
            return asst

def get_or_create_thread( thread_id = 'thread_cZLk7hjIlsR1uhGYttIAG2T9' ):
    if not thread_id:
        thread = client.beta.threads.create()
    else:
        thread = client.beta.threads.retrieve(thread_id)
    return thread


z = ZWrapper()
z.build_tree()
#z.print_tree()
zcol = z.get_collection('M5EN26AJ')
if zcol:
    zcol.read_items()
    #zcol.print_items()
    #zcol.print_item_tree()
    #for item in zcol.item_tree:
    #    print(item._item['data']['key'])

st.title('Zotero client')
st.write(zcol._collection['data']['name'], zcol._collection['data']['version'])
st.write(zcol.zot.last_modified_version())
for zitem in zcol.item_tree:
    if zitem._item['data']['itemType'] == 'journalArticle':
        st.write(zitem._item['data']['title'], "("+zitem._item['data']['key']+")", zitem._item['data']['version'])
        
        for child in zitem.child_item_list:
            #st.write(child)
            if child._item['data']['itemType'] == 'attachment':
                st.write(child._item['data']['filename'], child._item['data']['version'])

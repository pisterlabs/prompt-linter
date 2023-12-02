# Copyright (c) 2023, Xurpas AI Lab and contributors
# For license information, please see license.txt

import frappe,os,openai,requests
from frappe.utils import get_site_name
from frappe.model.document import Document
from frappe.utils import cstr

from langchain.llms import OpenAI
from llama_index import StorageContext, load_index_from_storage

from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext

from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


class XDADocument(Document):
	pass


@frappe.whitelist()
def create_thumbnail(doc,document,directory_path):
    name = doc.name

    # Thumbnail
    # document='./'+get_site_name(frappe.local.request.host)+document
    new_path='./'+get_site_name(frappe.local.request.host)+'/public/files/xda/thumbnails'
    os.makedirs(new_path, exist_ok=True)
    path='/files/xda/thumbnails/'+name+'.png'
    doc.thumbnail_image=path
    print(f'THUMBNAIL={doc.thumbnail_image}')

    convert_to_png(document,'./'+get_site_name(frappe.local.request.host)+'/public'+path)
    # Remove Existing Index
    files = []
    files.append(directory_path+'/docstore.json')
    files.append(directory_path+'/graph_store.json')
    files.append(directory_path+'/vector_store.json')
    files.append(directory_path+'/index_store.json')
    for item in files:
        try:
            print(f'REMOVING={item}')
            os.remove(item)
        except:
            pass

    # Create Index
    api_key()
    max_input_size = 4096
    num_outputs = 256
    max_chunk_overlap = .9
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-4", max_tokens=num_outputs))
    print('LOADING DOCS FROM DIRECTORY')
    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    print('INDEXING')
    index =  GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    print('SAVING INDEX TO DISK')
    index.storage_context.persist(directory_path)



@frappe.whitelist()
def move_document(name,document,target_path):
    filename=document[document.rfind('/')+1:]
    print(f'FILENAME={filename}')
    
    document='./'+get_site_name(frappe.local.request.host)+document
    target_path='./'+get_site_name(frappe.local.request.host)+target_path
    print(f'DOCUMENT={document}, TARGET={target_path}')
    doc = frappe.get_doc('XDA Document',name)
    if doc.name1== None or doc.name1 == '':
        doc.name1=filename
    from_path='/private/files/'
    new_path='./'+get_site_name(frappe.local.request.host)+'/private/files/xda/'+name+'/doc'
    print(f'DOCUMENT={document}, NEW_PATH={new_path}')
    doc.thumbnail=target_path
    doc.path=new_path
    print(f'PATH={target_path}')
    os.makedirs(new_path, exist_ok=True)
    try:
        os.rename(document,target_path)
    except:
        pass

    create_thumbnail(doc,target_path,new_path)
    doc.status='INDEXED'
    doc.save()
    frappe.db.commit()




def convert_to_png(file_path,savefile):
    images = convert_from_path(file_path)
    for item in images:
        print(item)
        item.save(savefile)
        return


def api_key():
    os.environ["OPENAI_API_KEY"] = frappe.get_doc('XDA OpenAI Settings').openai_api_key
    openai.api_key= os.environ["OPENAI_API_KEY"]


    

import requests
import json
import time
import os

class SynthesiaVideoAPI:
    def __init__(self, api_key):

        self.api_key = api_key

        self.headers = {
            "Authorization": f"{self.api_key}",
            "Content-Type": "application/json"
        }

        self.base_url = "https://api.synthesia.io/v2/videos"

    def create_video_from_template(self, template_id, template_data, title=None, description=None, visibility=None, test=None, callback_id=None):
        
        url = f"{self.base_url}/fromTemplate"
        print(f'URL={url}')
        print(f'HEADERS={self.headers}')
        
        data = {
            "templateId": template_id,
            "templateData": template_data,
        }

        if test is not None:
            data["test"] = test
        if title is not None:
            data["title"] = title
        if description is not None:
            data["description"] = description
        if visibility is not None:
            data["visibility"] = visibility
        if callback_id is not None:
            data["callbackId"] = callback_id
        # jsonData=json.loads(data)

        print(f'DATA={json}')

        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            return f'ERROR: {errh}'
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
            return f'ERROR : {errc}'
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
            return f'ERROR: {errt}'
        except requests.exceptions.RequestException as err:
            print ("Something went wrong",err)
            return f'ERROR: {err}'

        return response.json()

    def retrieve_video(self, video_id):

        url = f"{self.base_url}/{video_id}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print ("HTTP Error:",errh)
            return None
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
            return None
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
            return None
        except requests.exceptions.RequestException as err:
            print ("Something went wrong",err)
            return None

        return response.json()

    def download_video(self, video_url, output_path):
        response = requests.get(video_url, stream=True)

        with open(output_path, 'wb') as out_file:
            out_file.write(response.content)


@frappe.whitelist()
def generate_summary(name,directory_path):
    api_key()
    print(f'GENERATING VIDEO FOR {name}')
    maindoc = frappe.get_doc('XDA Document',name)
    prompt='You are tasked to summarize a particular document. You are given the full details of the document in the context. Give a 200 word summary of the given document. Convey your thoughts with confidence and authority and in an instructive manner. Refer to the document as "This document"'
    index_path=directory_path
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    #index = GPTVectorStoreIndex.load_from_disk('/home/andy/frappe-bench/apps/aimgpt/aimgpt/private/')
    index = load_index_from_storage(storage_context)

    # GPT-4 Service Context

    llm = 'gpt-4'
    gpt_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name=llm))

    # define prompt helper
    # set maximum input size
    max_input_size = 8191
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 0.5
    gpt_prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=gpt_predictor, prompt_helper=gpt_prompt_helper)
   
    response = index.as_query_engine().query(prompt)
    print(f'SUMMARY={response.response}')
    maindoc.summary=response
    maindoc.status='SUMMARIZED'
    maindoc.save()
    frappe.db.commit()


@frappe.whitelist()
def generate_medrep_spiel(name,directory_path):
    api_key()
    print(f'GENERATING SPIEL FOR {name}')
    maindoc = frappe.get_doc('XDA Document',name)
    prompt='You are a sales representative tasked to sell the product in the given document. You are given the full details of the product document in the context. Give a 200 word pitch of the product in the document. Convey your thoughts with confidence and authority.'
    if (maindoc.organization_name != None and maindoc.organization_name != ''):
        prompt=f'You are a sales representative for the company {maindoc.organization_name},tasked to sell the product in the given document. You are given the full details of the product document in the context. Give a 200 word pitch of the product in the document. Convey your thoughts with confidence and authority.'
    
    index_path=directory_path
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    #index = GPTVectorStoreIndex.load_from_disk('/home/andy/frappe-bench/apps/aimgpt/aimgpt/private/')
    index = load_index_from_storage(storage_context)

    # GPT-4 Service Context

    llm = 'gpt-4'
    gpt_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name=llm))

    # define prompt helper
    # set maximum input size
    max_input_size = 8191
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 0.5
    gpt_prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=gpt_predictor, prompt_helper=gpt_prompt_helper)
   
    response = index.as_query_engine().query(prompt)
    print(f'SUMMARY={response.response}')
    maindoc.summary=response
    maindoc.status='SUMMARIZED'
    maindoc.save()
    frappe.db.commit()

@frappe.whitelist()
def check_video_status(video_id):
    settings = frappe.get_doc('XDA OpenAI Settings')
    api = SynthesiaVideoAPI(settings.synthesia_api_key)
    response = api.retrieve_video(video_id)
    status=response['status']
    frappe.msgprint(
        msg=f'Status={status}',
        title='Video Check',
    )



@frappe.whitelist()
def generate_video(name):
    print(f'GENERATING VIDEO FOR {name}')
    maindoc = frappe.get_doc('XDA Document',name)
    settings = frappe.get_doc('XDA OpenAI Settings')
    api = SynthesiaVideoAPI(settings.synthesia_api_key)
    template_data = {
        "scene1Text": maindoc.summary,
        "thumbnail": 'https://'+get_site_name(frappe.local.request.host)+maindoc.thumbnail_image
    }
    print(f'DATA {template_data}')
    # try:
    create_response = api.create_video_from_template(
                settings.synthesia_template_id, # template id
                template_data,                          # template data
                "XDA Video",                        # title
                "XAIL Document Assistant Generated Video",            # description
                "private",                              # visibility
                True,                                   # test
                None                                    # callback
            )
    print(f'RESPONSE {create_response}')
    if isinstance(create_response,str):
        return create_response
    
    if create_response is not None:
        print(json.dumps(create_response, indent=4))

        video_id = create_response['id']
        maindoc.video_id = video_id
        maindoc.status='GENERATING_VIDEO'
        maindoc.save()
        print(f'VIDEO ID {video_id}')
        return
    return
    # except Exception as e:
    #     return f'ERROR GENERATING VIDEO: {e}'



def retrieve_videos():
    settings = frappe.get_doc('XDA OpenAI Settings')
    api = SynthesiaVideoAPI(settings.synthesia_api_key)
    doclist=frappe.db.get_list('XDA Document',filters={
        'status': 'GENERATING_VIDEO'
        },
        fields=['name'])
    
    for item in doclist:
         print(f"CHECKING {item.name}")
         thisdoc = frappe.get_doc('XDA Document',item.name)
         if (thisdoc.video_id != None and thisdoc.video_id !=''):
              video = api.retrieve_video(thisdoc.video_id)
              print(f"VIDEO {video}")
              status = video['status']
              if (status == 'complete'):
                download= video['download']
                print(f"DOWNLOAD {download}")
                path='./'+cstr(frappe.local.site)+'/public/files/xda/videos'
                os.makedirs(path, exist_ok=True)
                path=path+'/'+item.name+'.mp4'
                thisdoc.video_location=path
                thisdoc.video_url = download
                thisdoc.video_path='/files/xda/videos/'+item.name+'.mp4'
                thisdoc.status='DONE'
                api.download_video(download,path)
                thisdoc.save()
                frappe.db.commit()




# Template ID 70bac61e-700d-4cc0-a573-efea18c69a69
# # Usage:
# api = SynthesiaVideoAPI('') # API key

# template_data = {
#     "video_title": "Xurpas AI Lab",
#     "scene2Text": "Welcome to Xurpas AI Lab",
#     "precisionImage": "https://trami-data-folder.s3.ap-southeast-1.amazonaws.com/ai_lab_data/accuracy_graph.png",
#     "precisionText": "The graph indicates the training and test accuracy learning curve. An accuracy learning curve graph visually represents the model's learning process in terms of classification accuracy. The training accuracy converges to a value of 0.0055. The validation accuracy converges at 0.001. There is a gap between training and validation accuracies, indicating room for improvement in generalization to enhance performance on test data.",
#     "lossImage": "https://trami-data-folder.s3.ap-southeast-1.amazonaws.com/ai_lab_data/loss_graph.png",
#     "lossText": "This graph shows two curves, the training and test curve. They show how the model’s error changes as it iteratively processes and learns from data over multiple training cycles. The training loss starts at a high of 0.5 and converges at 0.3 whereas the test loss starts at 0.3 and converges at 0.1. The high starting value at the start of training indicates the models predictions are far from the actual value. While training progresses, the loss gradually decreases, indicating that the model is getting better at making predictions that align with the actual data.",
#     "trainingPrecision": "0.32934",
#     "trainingRecall": "0.339415",
#     "trainingF1Score": "0.229214",
#     "testPrecision": "0.368273",
#     "testRecall": "0.445026",
#     "testF1Score": "0.256374",
#     "scene5Text": "The results, with a train, and test recall of 0.32934 and 0.339415 respectively. and a train and test precision of 0.368273 and 0.229214, indicate that the model, trained on a highly imbalanced dataset places a significant emphasis on recall. This suggests that the model is effective at identifying positive (hired) instances, which can be valuable depending on the application's context. However, it's important to consider the trade-off between missing positive instances and producing false positives, as precision is slightly lower. The F1-Score of something indicates a balance between precision and recall, though further optimization may be possible."
# }

# create_response = api.create_video_from_template
# (
#     "55a7adb7-30aa-450c-8c59-059f4e399e85", # template id
#     template_data,                          # template data
#     "Xurpas AI Lab",                        # title
#     "This is a Synthesia video",            # description
#     "private",                              # visibility
#     True,                                   # test
#     None                                    # callback
# )

# if create_response is not None:
#     print(json.dumps(create_response, indent=4))

#     video_id = create_response['id']

#     # Wait for the video to be ready
#     while True:
#         retrieve_response = api.retrieve_video(video_id)

#         if retrieve_response is not None and retrieve_response['status'] == 'complete':
#             print(json.dumps(retrieve_response, indent=4))
#             filename = retrieve_response['title']

#             # Download the video
#             video_url = retrieve_response['download']
#             output_path = os.path.join(os.getcwd(), f'{filename}.mp4')
            
#             api.download_video(video_url, output_path)
            
#             print(f'Video downloaded to {output_path}')
            
#             break

#         print('Video is not ready yet. Waiting for 5 minutes before checking again.')
#         time.sleep(300)  # Wait for 5 minutes before checking again

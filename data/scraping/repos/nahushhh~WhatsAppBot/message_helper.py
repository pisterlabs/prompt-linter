import requests
import json
import os
from flask import request
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
API_KEY = "your_key"
class WhatsAppWraper:

    ## loading the config.json file
    with open('config.json', 'r') as json_file:
            data = json.load(json_file)  
    
    ## loading the corpus
    loader = UnstructuredFileLoader('content.txt')
    raw_documents = loader.load()

    ## splitting into chunks
    r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=20,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
    docs = r_splitter.split_documents(raw_documents)

    ## creating embeddings and storing them in a vectorstore
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)

    def __init__(self):   
        self.header = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {self.data['ACCESS_TOKEN']}"
        }
        

    ## sending template message
    def send_template_msg_function(self):
        template_url = f"https://graph.facebook.com/{self.data['VERSION']}/{self.data['PHONE_NUMBER_ID']}/messages"
        print(template_url)

        msg_template_params = {
            "messaging_product": "whatsapp",
            "to": f"{self.data['RECIPIENT_WAID']}",
            "type": "template",
            "template": {
                "name": "hello_world",
                "language": {
                    "code": "en_US"
                }
            }
        }
        response = requests.post(template_url, json=msg_template_params, headers=self.header)
        return response

    # send text message
    def send_text_msg_function(self, body):
        txt_url = f"https://graph.facebook.com/{self.data['VERSION']}/{self.data['PHONE_NUMBER_ID']}/messages"
        print(txt_url)

        text_msg_params = {
            "messaging_product": "whatsapp",    
            "recipient_type": "individual",
            "to": f"{self.data['RECIPIENT_WAID']}",
            "type": "text",
            "text": {
                "preview_url": False,
                "body": body
            }
        }

        text_response = requests.post(url=txt_url, json=text_msg_params, headers=self.header)
        return text_response
    
    def webhook_handler(self):
        res = request.get_json()
        print(res)
        print()
        user_res = ""

        ## to get the user's response or question
        if res['entry'][0]['changes'][0]['value']['messages'][0]['id']:
                user_res += res['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
                user_res = user_res.lower()
                # print(user_res)
        
        # user_res += res['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
        # print(user_res)
        
        ## querying 
        def ask_question(question):
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            retriever = self.vectorstore
            qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever.as_retriever(), memory = memory)
            result = qa_chain({"question": question})
            return result['answer']

        ans = ask_question(user_res)
            
        return ans


# def process_webhook_notification(data):
        
#         """_summary_: Process webhook notification
#         For the moment, this will return the type of notification
#         """
#         return data
    
    # if text_response.status_code==200:
    #     print("Sent message successfully!")
    #     # return text_response.json()
    # else:
    #     print("Error in sending message")
    #     # return text_response.json()





# text_result = send_text_message("Hey how are you!")
# print(text_result)



# async def send_message(data):
#   headers = {
#     "Content-type": "application/json",
#     "Authorization": f"Bearer {current_app.config['ACCESS_TOKEN']}",
#     }
  
#   async with aiohttp.ClientSession() as session:
#     url = 'https://graph.facebook.com' + f"/{current_app.config['VERSION']}/{current_app.config['PHONE_NUMBER_ID']}/messages"
#     try:
#       async with session.post(url, data=data, headers=headers) as response:
#         if response.status == 200:
#           print("Status:", response.status)
#           print("Content-type:", response.headers['content-type'])

#           html = await response.text()
#           print("Body:", html)
#         else:
#           print(response.status)        
#           print(response)        
#     except aiohttp.ClientConnectorError as e:
#       print('Connection Error', str(e))

# def get_text_message_input(recipient, text):
#   return json.dumps({
#     "messaging_product": "whatsapp",
#     "preview_url": False,
#     "recipient_type": "individual",
#     "to": recipient,
#     "type": "text",
#     "text": {
#         "body": text
#     }
#   })

# def get_templated_message_input(recipient):
#   return json.dumps({
#     "messaging_product": "whatsapp",
#     "to": recipient,
#     "type": "template",
#     "template": {
#         "name": "hello_world",
#         "language": {
#             "code": "en_US"
#         }
#     }
# })
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import openai

import os
import re
import json
import jsonpickle
import sys

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from utils.get_credential import get_credentials, is_valid_token



from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain.chains.conversation.memory import ConversationEntityMemory, ConversationBufferWindowMemory, ConversationBufferMemory
# from langchain.memory import ConversationBufferMemory
# from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from utils.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains import ConversationChain

from typing import Optional


from utils.database import init_app, db_sqlalchemy, app
from utils.database import User as User
from utils.database import Message as Message
from utils.database import write_chat_to_db
from utils.whatsapp_agent import predict_gpt

from .whatsapp_utils import audio_to_text, get_location_from_message,recognize_image_from_url, analyze_video, analyze_document

from pprint import pprint
import streamlit as st



from dotenv import load_dotenv  
load_dotenv('.credentials/.env')


# Initialize Phone Number

phone_number = ""
if "phone_number" not in st.session_state:
  st.session_state["phone_number"] = phone_number




# Fungsi untuk mengirim pesan WhatsApp menggunakan API Wablas
def send_whatsapp_message(phone_number, message):
    url = "https://pati.wablas.com/api/v2/send-message"
    token = os.environ['WABLAS_TOKEN']  # Mengambil token dari environment variable
    headers = {"Authorization": token, "Content-Type": "application/json"}
    payload = {"data": [{"phone": phone_number, "message": message}]}

    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    response = requests.post(url,
                     headers=headers,
                     data=json.dumps(payload),
                     verify=False, 
                     timeout=30)
    
    result = response.json()

    return result






# Fungsi untuk menangani pesan masuk dari webhook
def handle_incoming_message(data):
    credentials = None

    print(f"Data Masuk: {data}")

    try:
        phone = data.get('phone', None)  # Mengambil nomor telepon pengirim pesan
        incoming_message = data.get('message', None)  # Mengambil isi pesan
        isFromMe = data.get('isFromMe', None)
        sender = data.get('sender', None)
        user_name = data.get('pushName', None)
        messageType = data.get('messageType', None)
        url = data.get('url', None) 

        message = ''


        data_str = ""
        for key, value in data.items():
            if value not in [None, ""]:
                data_str += f"{key}: {value}" + "\n"  
        
        print(f'\n\nPesan masuk dari {phone}: {incoming_message} (isFromMe: {isFromMe})\n\n')

        if isFromMe:
            print('Pesan dikirim oleh bot. Tidak perlu direspon')
            exit()

        #chek apakah nomor telepon sudah ada di database
        with app.app_context():
            user = User.query.filter_by(phone_number=phone).first()

            if user:
                msg = ""
                #lakukan pengecekan apabila format pesan masuk sesuai dengan format token
                if re.match(r'^[0-9a-fA-F]{64}$', incoming_message):
                    #Check apakah token valid ketika pesan masuk sama dengan token yang ada di database
                    
                    credentials = get_credentials(incoming_message)
                    print(f'Credentials: {credentials}')    
                    print(f"Format sesuai? {re.match(r'^[0-9a-fA-F]{64}$', incoming_message)}")

                    if credentials is not None:

                        url = credentials['url']
                        username = credentials['username']
                        password = credentials['password']
                        created_at = credentials['created_at']
                        mobile_phone = credentials['mobile_phone']
                        
                        # Calculate remaining time
                        remaining_time = created_at + timedelta(days=5) - datetime.now()
                        remaining_hours = remaining_time.total_seconds() // 3600

                        msg += f"""Terimakasih Token anda sudah diverifikasi.\n\nURL: {url}\nUsername: {username}\nPassword: ***\nMobile Phone: {mobile_phone}\nCreated At: {created_at}\nToken will expire in {remaining_hours} hours"""
                        msg += "\n\nApakah ada yang bisa saya bantu?"            

                    else:
                        msg += "Invalid token. Please check your token."


                    message = msg
                    send_whatsapp_message(phone, message)  # Mengirim respon ke pengirim pesan
                    print(f'Pesan hasil pengecekan Token: {msg}')

                    exit()

            
            else:
                user = User(phone_number=phone, username=user_name)
                db_sqlalchemy.session.add(user)
                db_sqlalchemy.session.commit()

        # Check type pesan kemuadian lakukan proses sesuai dengan type pesan
        if messageType == "audio":
            incoming_message = audio_to_text(url)
            message = f"_Audio:[{incoming_message}]_\n\n"
        elif messageType == "location":
            incoming_message = get_location_from_message(incoming_message)
            message = f"_[{incoming_message}]_\n\n"
        elif messageType == "image":
            images_info = recognize_image_from_url(url)
            print(f'Incoming Message: {incoming_message}')


            base_prompt_1 = f'''
            - Berikut adalah keterangan yang bisa digunakan sebagai materi untuk menyusun respons: {str(images_info)}.
            - "Text dalam gambar" diperoleh dari OCR foto. Prediksi teks terpotong dari konteks.
            '''
            base_prompt_2 = f'''
            - Gunakan "sepertinya", "mungkin", "saya rasa" untuk hal tidak pasti.
            - Susun kalimat motivasi dengan kata-kata di atas.
            - Tambahkan ekspresi dan emoticon relevan di akhir. 
            '''

            if incoming_message == '' or incoming_message is None:
                incoming_message = f'\nRespon dengan satu kalimat ekspresif, sesuai dengan catatan sebagai berikut:{base_prompt_1} {base_prompt_2}'
            else:
                incoming_message = f'''Caption:{incoming_message}\n---\nRespon sesuai dengan caption diatas, dengan catatan sebagai berikut:{base_prompt_1}'''

              
        elif messageType == "video":
            incoming_message = analyze_video(url)
            # message = f"_Video Description: {video_description}_\n\n"
        elif messageType == "document":
            incoming_message = analyze_document(url)
        else:
            message = ""



        # Persiapkan pesan untuk dikirimkan ke whatsapp
        # if phone ==  "628112227980": #Selama masa percobaan, hanya nomor ini yang bisa mengakses
        if phone and incoming_message is not None:    
            # message = message + prepare_message(phone, incoming_message)

            response = predict_gpt(phone, incoming_message)
            print(f'Response: {response}')

            output = response['output']
            total_cost = response['total_cost']


            

            message = message + str(output) + f' [{str(total_cost)}]'

            send_whatsapp_message(phone, message)  # Mengirim respon ke pengirim pesan
            write_chat_to_db(user_name, phone, incoming_message, sender, message, total_cost)


            
        return jsonify({'status': 'success', 'phone': phone})

    except Exception as e:
        if(phone == "628112227980"):
            error_msg = f"Error (handle_incoming_message): {e}\n\n"
            print(error_msg)
        else:
            error_msg = ""

        message = prepare_message(phone, incoming_message)
        send_whatsapp_message(phone, message)
        # # send_whatsapp_message('628112227980', f"{error_msg}Kirimkam 'reset' untuk merefresh percakapan baru.")
        raise
 

# Fungsi untuk mendapatkan respon dari model chatgpt
def prepare_message(phone, incoming_message):
    session_state = {
    'past': [],
    'generated': []
    }


    #Apabila incoming_message diawali dengan "GPT4/" maka gunakan model GPT4 dengan openai_api_key yang terpisah
    K = 5  #Jumlah Histori yang perlu di konsider
    if incoming_message.startswith("GPT4/"):
        print("Menggunakan model GPT4")
        MODEL = 'gpt-4'
        API_O = os.environ['OPENAI_KEY_GPT4']
        #buang "GPT4/" dari incoming_message
        incoming_message = incoming_message[5:]
    else:
        print("Menggunakan model GPT3")
        MODEL = 'gpt-3.5-turbo'
        API_O = os.environ['OPENAI_KEY']


    # Session state storage would be ideal
    if API_O:
        # Create an OpenAI instance
        llm = ChatOpenAI(temperature=0,
                    openai_api_key=API_O,
                    model_name=MODEL,
                    verbose=True)

        #Membaca Entity Memory dari database        
        with app.app_context():
            user_query = User.query.filter_by(phone_number=phone).first()
            
            if user_query is not None:
                buf_memory_json = user_query.entity_memory
            else:
                buf_memory_json = None

        #Reset Entity Memory ketika pesan masuk adalah "reset"
        if incoming_message.lower() == "reset":
            print('\n\nMelakukan reset memori...\n\n')
            buf_memory_json = None
            incoming_message = "Katakan 'Memori percakapan sebelumnya sudah saya hapus.'"


        if buf_memory_json is None:
            # We set a low k=2, to only keep the last 2 interactions in memory
            #memory = ConversationBufferWindowMemory(k=K)
            # memory = ConversationEntityMemory(llm=llm, k=K)

            memory = ConversationBufferMemory(memory_key="chat_history")
            # print('\n\nGetMemory from DB: Tidak ditemukan memory dalam DB\n\n')

        else:
            memory = jsonpickle.decode(buf_memory_json)
            #print(f'\n\nGetMemory from DB: {memory}\n\n')
    

        # Create the ConversationChain object with the specified configuration
        Conversation = ConversationChain(llm=llm,
                                        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                                        memory=memory,
                                        verbose=True)
    

        with get_openai_callback() as cb:
            output = Conversation.run(input=incoming_message)
            print(f'\nOutput: {output}\n')
    
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (IDR): IDR {cb.total_cost*15000}\n")


        #Menyimpan Entity Memory ke database
        buf_memory = Conversation.memory #sebelumnya sampe memory doang
        
        with app.app_context():
            user_query = User.query.filter_by(phone_number=phone).first()
            if user_query is None:
                print(f"No user found with phone: {phone}")
            else:
                print(f"User found with phone: {phone}")
                try:
                    # st.session_state["entity_memory"] = buf_memory #save entity memory to session state

                    buf_memory_jsonpickle = jsonpickle.encode(buf_memory)
                    user_query.entity_memory = buf_memory_jsonpickle
                    db_sqlalchemy.session.commit()
                    print(f"\n\nSuccessfully updated entity memory for user with phone: {phone}")
                except Exception as e:
                    print(f"\n\nFailed to update entity memory: {str(e)}")
    
    return output
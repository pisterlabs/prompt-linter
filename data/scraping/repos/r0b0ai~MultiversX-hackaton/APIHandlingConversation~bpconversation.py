from flask import Blueprint,Flask, request, jsonify, current_app, session
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from datetime import datetime,timedelta, timezone
import unicodedata
import openai
import traceback
import sentry_sdk
from sentry_sdk import set_user
from sentry_sdk.integrations.flask import FlaskIntegration
import random
import threading
import json
import pytz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm import joinedload
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from extensions import db
from models import *
import requests as r
from db_helpers import *
from recognition_manager import RecognitionManager
from recognition_manager_rasa import RecognitionManagerRasa
from flask_cors import CORS
from flask import Flask, jsonify, request
from sqlalchemy.orm.exc import NoResultFound
import json
import base64
from werkzeug.utils import secure_filename
import os
from flask_jwt_extended import create_access_token,get_jwt,get_jwt_identity, \
                               unset_jwt_cookies, jwt_required, JWTManager
import stripe
import hashlib
import requests
import pickle
from contextlib import contextmanager
import requests
from random import choice


Bpconversation = Blueprint('bp_conversation', __name__, template_folder="conversation")
RASA_URL = ''

# Initialize the OpenAI API with your key
openai.api_key = 'OPEN AI API KEY'

# The library needs to be configured with your account's secret key.
endpoint_secret = 'YOUR SECRET KEY'
stripe.api_key = 'STRIPE API KEY'

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = db.session
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def sendToTranslator_en_sl(sentence, url):

    headers = {
        'Content-Type': 'text/plain;charset=UTF-8'
    }
    response = r.request("POST", url, headers=headers, data=sentence.encode('utf-8'))
    print("------",response.json())

    data = {
        'status': 'true',
        'besedilo': response.json()["besedilo"]
    }
    return json.dumps(data)



def replace_urls_with_links(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(lambda x: '<a href="{}">{}</a>'.format(x.group(), x.group()), text)

conversation_states = {}
def process_dialogue_rule(app,rule_name, rule_data, conversation_states, sender, text, response_text, gesture, service_id, email):
    with app.app_context():
        required_state = rule_data.get("required_state", "initial")
        responses = rule_data.get('response', [])
        if not responses:
            print("No responses available for rule:", rule_name)
            return  # Or handle it differently if needed
        recognition_manager = RecognitionManager(sender, service_id, email)
        # Get the predicted intent of the user's text
        category = recognition_manager.get_best_match_category(text)
        if conversation_states[sender] == required_state and rule_name == category:
            
            random_response = choice(responses)
            print("----------------",replace_urls_with_links(random_response))
            #print("kaj dobim kot response", random_response)
            response_text[0] = replace_urls_with_links(random_response)
            gesture[0] = rule_data["gesture"]
            print("tukaj se 1", rule_data)

            if "next_state" in rule_data:
                conversation_states[sender] = rule_data["next_state"]
            else:
                conversation_states[sender] = "initial"



def get_robot_module(email, robot_name, service_id):


    # Endpoint URL
    url = "http://153.5.66.230:5000/get_robot_module"

    # Parameters you want to send
    params = {
        'email': email,
        'robot_name': robot_name,
        'service_id': service_id
    }


    # Make the GET request
    response = requests.get(url, params=params)

    # Return the parsed JSON response
    return response.json()



def check_stripe_subscription(customer_email, module_name):
    products = stripe.Product.list(limit=80)  # Adjust the limit as per your needs

    # Fetch the customer from Stripe
    customers = stripe.Customer.list(email=customer_email)
    if not customers['data']:
        return False  # Return False if customer not found

    # Get all active subscriptions of the customer from Stripe
    stripe_subscriptions = stripe.Subscription.list(customer=customers['data'][0].id, status='active')

    for product in products:
        is_subscribed = any(
            item['price']['product'] == product.id
            for subscription in stripe_subscriptions['data']
            for item in subscription['items']['data']
        )
        if is_subscribed and product.name == module_name:
            print("pa je res kaj true tule----------------", product.name)
            return True  # User is subscribed to the provided module name
    return False  # User is not subscribed to the provided module name


@Bpconversation.route('/set_session')
def set_session():
    session['is_subscribed_to_fizika'] = True
    session.modified = True
    return "Session set!"

@Bpconversation.route('/get_session')
def get_session():
    return str(session.get('is_subscribed_to_fizika', None))



def email_to_filename(email):
    # Use SHA256 to get a fixed-size string that doesn't contain invalid characters
    return hashlib.sha256(email.encode()).hexdigest() + '.pkl'

def get_module_type_by_name(module_name):
    # Query the module using the provided name
    module = Modules.query.filter_by(name=module_name).first()
    
    # If the module exists, return its type. Otherwise, return None.
    return module.type if module else None


def sendToRasa(name, data, service_id):
    menu, text = '', ''
    gestures, music, image = [], [], []
    port = None

    result = {'menu': None, 'text': None, 'gestures': None, 'music': None, 'port': None, 'image': None, 'media': None, 'gender': None, 'language_code': None}

    payload = json.dumps({
        "sender": name,
        "message": data
    })
    headers = {
        'Content-Type': 'application/json'
    }

    user_details = getMailPasswordFromUser(name, service_id)
    if not user_details:
        # Log an error or handle the absence of user details as you see fit
        print(f"No user details found for sender: {name}")
        return None  # or handle appropriately
    
    email = user_details["name"]
    password = user_details["password"]


    # JWT token (this should be dynamically fetched when the user logs in)
    jwt_token = LoginUserChatbot(email,password)['access_token']

    headers = {
        'Authorization': f'Bearer {jwt_token}'
    }

    gender = getGenderLanguage(jwt_token ,name, service_id)["gender"]
    language = getGenderLanguage(jwt_token ,name, service_id)["language_id"]

    response = r.request("POST", RASA_URL, headers=headers, data=payload)
    print("dobil sem od rase", response.text)
    print("kaj je to", type(response.json))

    response_json = response.json()

    if len(response_json) > 0:
        media_id = None

        custom_data = response_json[0].get('custom', None)
        if isinstance(custom_data, dict):
            media_id = custom_data.get('media_id', None)
        elif isinstance(custom_data, list):
            for item in custom_data:
                if 'media_id' in item:
                    media_id = item['media_id']
                    break
    
        print(f"Media ID is: {media_id}")  # Debugging line to check media_id value

        if media_id is not None:
            media = Media.query.get(media_id)  # Get the media object by its ID from the database
            print(f"Media Object is: {media}")  # Debugging line to check media object
            
            if media:
                # Encode the binary media file as a base64 string
                encoded_file = base64.b64encode(media.file).decode('utf-8')
                result['media'] = {
                    "type": media.type,
                    "file": encoded_file
                }
            else:
                result['media'] = None
        else:
            result['media'] = None


        if isinstance(custom_data, list):
            for item in custom_data:
                if 'text' in item:
                    text = item['text']
                if 'gestures' in item.get('custom', {}):
                    gestures.extend(item['custom']['gestures'])
                if 'port' in item.get('custom', {}):
                    port = item['custom']['port']
                    break
        elif isinstance(custom_data, dict):
            text = custom_data.get('text', '')
            if 'gestures' in custom_data:
                gestures.extend(custom_data['gestures'])
            if 'port' in custom_data:
                port = custom_data['port']

        if 'custom' in response_json[0] and 'port' in response_json[0]['custom']:
            port = response_json[0]['custom']['port']
        else:
            try:
                port = response_json[1]['custom']['port']
            except:
                try:
                    port = response_json[0]['custom']['port']
                except:
                    try:
                        menu = 'chat' in response_json[0]['custom']
                    except:
                        menu = 'chat' in response_json[1]['custom']

    #result = {'menu': str(menu), 'text': text, 'gestures': gestures, 'music': music, 'port': port, 'image': image, 'gender': gender, 'language_code': language}
    result.update({
    'menu': str(menu), 
    'text': text, 
    'gestures': gestures, 
    'music': music, 
    'port': port, 
    'image': image, 
    'gender': gender, 
    'language_code': language
    })

    del response, headers, payload, text, gestures, music, image
    return result

                   

def getGenderLanguage(jwt ,robot_name, service_id):

    jwt_token =  jwt

    headers = {
        'Authorization': f'Bearer {jwt_token}',
        'Content-Type': 'application/json'
    }

    data = {
        'service_id': service_id,
        'robot_name': robot_name
    }

    response = requests.post('http://153.5.66.230:5000/get_robot_details', headers=headers, json=data)

    robot_details = response.json()
    print("je tototototo",robot_details["gender"])

    return robot_details

def getMailPasswordFromUser(robot_name,service_id):
    BASE_URL = "http://153.5.66.230:5000"
    response = requests.get(f"{BASE_URL}/get_user_by_robot_name", params={"robot_name": robot_name, "service_id": service_id})

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def LoginUserChatbot(username, password):


        url = 'http://153.5.66.230:5000/login'

        # Request data (name and password)
        data = {
            'email': username,
            'password': password
        }

        # Convert the data to JSON
        json_data = json.dumps(data)

        # Set the headers for the request
        headers = {
            'Content-Type': 'application/json'
        }

        # Send the POST request to login the user
        response = requests.post(url, data=json_data, headers=headers)

        data = response.json()
        return data

def LoginUserChatbot(username, password):


        url = 'http://153.5.66.230:5000/login'

        # Request data (name and password)
        data = {
            'email': username,
            'password': password
        }

        # Convert the data to JSON
        json_data = json.dumps(data)

        # Set the headers for the request
        headers = {
            'Content-Type': 'application/json'
        }

        # Send the POST request to login the user
        response = requests.post(url, data=json_data, headers=headers)

        data = response.json()
        return data

# eučbeniki so tu koda
import requests

def query_api(query, subject, education, jwt_token):
    url = 'https://eučbeniki.aibc-robotics.com//search'
    params = {
        'query': query,
        'subject': subject,
        'education': education
    }
    headers = {
        'Authorization': f'Bearer {jwt_token}'
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()['result']
    else:
        print(f'Error: {response.status_code}')
        print(response.text)


# TODO dodaj redisa za chackanje requestov da bo šlo hitreje in bo notri v memory...

@Bpconversation.route('/search_keywords', methods=['POST'])
@jwt_required()
def search_keywords():
    with sentry_sdk.start_transaction(name="search_keyword", op="endpoint"):
        if not request.is_json:
            return jsonify({"error": "Request data must be JSON."}), 400

        input_data = request.get_json()

        if 'text' not in input_data or 'sender' not in input_data or 'service_id' not in input_data or 'email' not in input_data:
            return jsonify({"error": "Missing fields in JSON data."}), 400

        text = input_data['text'].lower()
        sender_name = input_data['sender']
        print("sender name------",sender_name)
        service_id = input_data['service_id']
        email = get_jwt_identity()

        # Get the client's IP address
        client_ip = request.remote_addr

        # Add the user's context to Sentry for enhanced error reporting
        set_user({
            "email": email,
            "ip_address": client_ip
            # You can add more fields here if needed
        })


        # Fetch the user by email first
        user = Sender_user.query.filter_by(name=email).first()
        if user is None:
            return jsonify({"error": "User not found."}), 404
        
        #stripe preverjanje and gettanje modula----------------------------------------------------

        result = get_robot_module(get_jwt_identity(), sender_name, service_id)


        filename = email_to_filename(email)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        with requests.Session() as s:
            session_exists = True
            
            # Try loading the serialized session cookie data if it exists
            try:
                with open(filename, 'rb') as f:
                    s.cookies.update(pickle.load(f))
            except FileNotFoundError:
                session_exists = False

            # If session doesn't exist, call the set_session endpoint
            if not session_exists:
                s.get('http://153.5.66.230:5000/set_session', headers=headers)
                with open(filename, 'wb') as f:
                    pickle.dump(s.cookies, f)
            
            # Now use the session
            response = s.get('http://153.5.66.230:5000/get_session', headers=headers)
            is_subscribed_to_fizika = response.text

        
        #----------------------------------------------------------------------

        # ------------ pridobi tip modula za naprej
        if user.has_free_access == 1 and int(service_id) == 3 and get_jwt_identity() == "obcina_dobrna@gmail.com" or user.has_free_access == 1 and int(service_id) == 3 and get_jwt_identity() == "marija.svent@dobrna.si" :
            print("tu sem free je")
            print(f"User {email} has free access. Skipping subscription checks.")
            print("kateri modul imam --------",  result.get("module"), sender_name)

            if  result.get("module"):
                module_type = get_module_type_by_name( result.get("module"))
            else:
                return jsonify({"error":"module is not provided"}), 400
            print("kateri modul imam tip", module_type)


            if module_type == "navadni":
                # Fetch the sender object by name, service_id and user id
                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user."}), 404

                # Extract the language and gender if the service_id is 1 or 2
                language_id = sender.language_id if service_id in [1, 2] else None
                gender = sender.gender if service_id in [1, 2] else None

                # Set default conversation state
                if sender not in conversation_states:
                    conversation_states[sender.name] = 'initial'


                recognition_manager = RecognitionManager(sender.name, service_id, email)
                if recognition_manager.check_exit_phrases(text):
                    print("tukaj sem zdaj reset")
                    conversation_states[sender.name] = 'initial'

                category = recognition_manager.get_best_match_category(text)
                print("kaj dobim tu", category)
                dialog_data = get_phrases_response_gesture_by_dialog_type(category, sender_name, email, service_id, result.get("module"))


                if "error" in dialog_data:
                    return jsonify({"error": dialog_data["error"]})

                response_text = ["Nisem razumel."]
                gesture = [""]
                threads = []

                t = threading.Thread(
                    target=process_dialogue_rule,
                    args=(current_app._get_current_object(),category, dialog_data, conversation_states, sender_name, text, response_text, gesture, service_id, email)
                )

                threads.append(t)
                t.start()

                # Wait for all threads to finish
                for t in threads:
                    t.join()   

                if isinstance(response_text, list) and len(response_text) == 1:  # ensure it's a list and has one item
                    sentences = re.split('(?<=[.!?]) +', response_text[0])
                    response_text = [sentence.strip() for sentence in sentences if sentence]
 

                print("----")
                print("response_text", type(response_text))
                response_data = {
                    "text": response_text,
                    "gestures": gesture,
                    "port": "50200",
                    "media": None  # Set media to None by default
                }
                print ("kak to zgleda", response_data)
                #dialog_id = get_dialog_id_by_type(category)  # Replace with the actual dialog ID
                dialog_id = get_dialog_id_by_type_and_sender(category, sender_name, email, service_id)

                print("to je ta", dialog_id)
                # Fetch the Dialog object with the matching dialog_id and sender_id
                dialog = Dialog.query.options(joinedload(Dialog.media)).filter_by(id=dialog_id, robot_id=sender.id).first()

                # Fetch associated media for the dialog
                if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    #print("Media file:", media.file)
                    
                    encoded_file = base64.b64encode(media.file).decode('utf-8')  # Convert bytes to base64-encoded string
                    response_data["media"] = {
                        "type": media.type,
                        "file": encoded_file
                    }
                    print("adsa kaj pa dobim tu", response_data)
                else:
                    print("No media found for the dialog.")
                #print ("kak to zgleda 2---", response_data)
                # TODO skumuniciraj z valentinom
                '''if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    

                    
                    response_data["media"] = {
                        "type": media.type,
                        "media_id": media.id,         # Provide media_id
                        "file_name": media.name,      # Provide file_name
                    }
                else:
                    print("No media found for the dialog.")'''


                # If the service_id is 1 or 2, add language and gender to the response
                if service_id in [1, 2]:
                    response_data["language_code"] = language_id
                    response_data["gender"] = gender

                return jsonify(response_data)
            
            if module_type == "demo":
                # Fetch the sender object by name, service_id and user id
                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user."}), 404

                # Extract the language and gender if the service_id is 1 or 2
                language_id = sender.language_id if service_id in [1, 2] else None
                gender = sender.gender if service_id in [1, 2] else None

                # Set default conversation state
                if sender not in conversation_states:
                    conversation_states[sender.name] = 'initial'


                recognition_manager = RecognitionManager(sender.name, service_id, email)
                if recognition_manager.check_exit_phrases(text):
                    print("tukaj sem zdaj reset")
                    conversation_states[sender.name] = 'initial'

                category = recognition_manager.get_best_match_category(text)
                print("kaj dobim tu", category)
                dialog_data = get_phrases_response_gesture_by_dialog_type(category, sender_name, email, service_id, result.get("module"))

                if "error" in dialog_data:
                    return jsonify({"error": dialog_data["error"]})

                response_text = ["Nisem razumel."]
                gesture = [""]
                threads = []

                t = threading.Thread(
                    target=process_dialogue_rule,
                    args=(current_app._get_current_object(),category, dialog_data, conversation_states, sender_name, text, response_text, gesture, service_id, email)
                )

                threads.append(t)
                t.start()

                # Wait for all threads to finish
                for t in threads:
                    t.join()   

                if isinstance(response_text, list) and len(response_text) == 1:  # ensure it's a list and has one item
                    sentences = re.split('(?<=[.!?]) +', response_text[0])
                    response_text = [sentence.strip() for sentence in sentences if sentence]
 

                print("----")
                print("response_text", type(response_text))
                response_data = {
                    "text": response_text,
                    "gestures": gesture,
                    "port": "50200",
                    "media": None  # Set media to None by default
                }
                print ("kak to zgleda", response_data)
                #dialog_id = get_dialog_id_by_type(category)  # Replace with the actual dialog ID
                dialog_id = get_dialog_id_by_type_and_sender(category, sender_name, email, service_id)

                print("to je ta", dialog_id)
                # Fetch the Dialog object with the matching dialog_id and sender_id
                dialog = Dialog.query.options(joinedload(Dialog.media)).filter_by(id=dialog_id, robot_id=sender.id).first()

                # Fetch associated media for the dialog
                if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    #print("Media file:", media.file)
                    
                    encoded_file = base64.b64encode(media.file).decode('utf-8')  # Convert bytes to base64-encoded string
                    response_data["media"] = {
                        "type": media.type,
                        "file": encoded_file
                    }
                    print("adsa kaj pa dobim tu", response_data)
                else:
                    print("No media found for the dialog.")
                #print ("kak to zgleda 2---", response_data)
                # TODO skumuniciraj z valentinom
                '''if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    

                    
                    response_data["media"] = {
                        "type": media.type,
                        "media_id": media.id,         # Provide media_id
                        "file_name": media.name,      # Provide file_name
                    }
                else:
                    print("No media found for the dialog.")'''


                # If the service_id is 1 or 2, add language and gender to the response
                if service_id in [1, 2]:
                    response_data["language_code"] = language_id
                    response_data["gender"] = gender

                return jsonify(response_data)

            if module_type == "custom":

                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user in custom module."}), 404

                #recognition_manager = RecognitionManager(sender.name, service_id, email)
                recognition_manager_rasa = RecognitionManagerRasa(sender.name, service_id, email)
                if recognition_manager_rasa.check_custom_phrases(text):
                    print("imam pravi text", sender_name, "----", text)
                    data_rasa = sendToRasa(sender_name,text, service_id)
                    print("je to to", data_rasa)
                    return data_rasa
                else:
                    return jsonify({"error":"dialog for custom module not in databases"})
            
            if module_type == "pro chatgpt":

                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user in chatgpt pro module."}), 404


                if sender:
                    print("kaj je text--", text)    
                  
                    # Define the URL of your Flask app
                    url = 'https://spletnastranchat.aibc-robotics.com/search'
                    user_details = getMailPasswordFromUser(sender.name,service_id)
                    if not user_details:
                        print(f"No user details found for sender: {sender.name}")
                        return None  # or handle appropriately
                    jwt = LoginUserChatbot(user_details["name"], user_details["password"])["access_token"]
                    # Define the headers for the request
                    headers = {
                        'Authorization': f'Bearer {jwt}'  # Make sure jwt_token is available in this scope
                    }

                    # Define the query parameter for the request
                    params = {
                        'query': text
                    }

                    response = requests.get(url, headers=headers, params=params)

                    if response.status_code == 200:
                        # Parse the JSON response
                        response_json = response.json()
                        search_result = response_json.get('result', '')  # Extract the result from the response
                    else:
                        print(f'Error: Unable to retrieve data. HTTP Status Code: {response.status_code}')
                        search_result = 'Error: Unable to retrieve data.'  # Default error message


                    # Construct the desired response shape
                    result = {
                        "gestures": ["tega ni"],  # example static gesture value
                        "media": None,       # example static media value
                        "port": "50200",     # example static port value
                        "text": [search_result]  # using the response from the external API
                    }

                    return jsonify(result)
                else:
                    return jsonify({"error": "dialog for custom module not in databases"})
            
            if module_type == "pro učbeniki":

                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user in chatgpt pro module."}), 404

                if sender:
                    print("kaj je text--", text, service_id)    

                    user_details = getMailPasswordFromUser(sender.name,service_id)
                    print("kaj dobim tu,", sender, user_details)
                    if not user_details:
                        print(f"No user details found for sender: {sender.name}")
                        return None  # or handle appropriately
                    jwt = LoginUserChatbot(user_details["name"], user_details["password"])["access_token"]

                    # Fetch subject and education (subject_id) from the database for sender
                    subject = sender.subject
                    education = sender.subject_id

                    if not subject or not education:
                        return jsonify({"error": "Subject or education information missing for the sender."}), 404

                    result_query_api = query_api(text, subject, education,jwt)


                    # Construct the desired response shape
                    result = {
                        "gestures": ["tega ni"],  # example static gesture value
                        "media": None,       # example static media value
                        "port": "50200",     # example static port value
                        "text": [result_query_api]  # using the response from the new query_api function
                    }

                    return jsonify(result)
                else:
                    return jsonify({"error": "dialog for custom module not in databases"})



            
            if module_type == "chatgpt":
                print("pridem notri chatgpt..--------------------")
                # Fetch the sender object by name, service_id and user id
                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user."}), 404

                # Extract the language and gender if the service_id is 1 or 2
                language_id = sender.language_id if service_id in [1, 2] else None
                gender = sender.gender if service_id in [1, 2] else None

                # Set default conversation state
                if sender not in conversation_states:
                    conversation_states[sender.name] = 'initial'


                recognition_manager = RecognitionManager(sender.name, service_id, email)
                if recognition_manager.check_exit_phrases(text):
                    print("tukaj sem zdaj reset")
                    conversation_states[sender.name] = 'initial'

                category = recognition_manager.get_best_match_category(text)
                print("kaj dobim tu", category)
                dialog_data = get_phrases_response_gesture_by_dialog_type(category, sender_name, email, service_id, result.get("module"))


                if "error" in dialog_data:
                    if dialog_data["error"] == 'No dialog found for the given dialog type. Please add dialogs or intents.':
                        # Call OpenAI API to get a response
                        refined_prompt = "Ti si pogovorni asistent, ki odgovarjaš na vprašanja in podajaš čimbolj pravilne odgovore. " + text
                        chat_gpt_response = openai.Completion.create(
                            engine="text-davinci-003",
                            prompt=refined_prompt,
                            max_tokens=150,
                            temperature=0.2
                        )
                        response = chat_gpt_response.choices[0].text.strip()

                        # Convert the response into a list of sentences/fragments
                        sentences = response.split('\n')

                        # Remove any standalone '!' from the list of sentences
                        sentences = [s for s in sentences if s != '!']

                        # Join the sentences into a single string
                        joined_response = ' '.join(sentences).strip()

                        # Remove any unwanted patterns (like HTML tags, if any are present)
                        cleaned_response = re.sub('<.*?>', '', joined_response)

                        # Append the note to indicate the response was generated by ChatGPT
                        final_response = cleaned_response + " (odgovor podal ChatGPT)"
                        response_text = [final_response]
                    else:
                        return jsonify({"error": dialog_data["error"]})


                response_text = ["Nisem razumel."]
                gesture = [""]
                threads = []

                t = threading.Thread(
                    target=process_dialogue_rule,
                    args=(current_app._get_current_object(),category, dialog_data, conversation_states, sender_name, text, response_text, gesture, service_id, email)
                )

                threads.append(t)
                t.start()

                # Wait for all threads to finish
                for t in threads:
                    t.join()
                print("response text-----", response_text)

                if not response_text or response_text == ["Nisem razumel."]:  # check if the response_text is not found or default
                    # Call OpenAI API to get a response
                    refined_prompt = "Ti si pogovorni asistent, ki odgovarjaš na vprašanja in podajaš čimbolj pravilne odgovore. " + text
                    chat_gpt_response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=refined_prompt,
                        max_tokens=150,
                        temperature=0.2
                    )
                    response = chat_gpt_response.choices[0].text.strip()

                    # Convert the response into a list of sentences/fragments
                    sentences = response.split('\n')

                    # Remove any standalone '!' from the list of sentences
                    sentences = [s for s in sentences if s != '!']

                    # Join the sentences into a single string
                    joined_response = ' '.join(sentences).strip()

                    # Remove any unwanted patterns (like HTML tags, if any are present)
                    cleaned_response = re.sub('<.*?>', '', joined_response)

                    # Append the note to indicate the response was generated by ChatGPT
                    final_response = cleaned_response + " (odgovor podal ChatGPT)"
                    response_text = [final_response]

                if isinstance(response_text, list) and len(response_text) == 1:  # ensure it's a list and has one item
                    sentences = re.split('(?<=[.!?]) +', response_text[0])
                    response_text = [sentence.strip() for sentence in sentences if sentence]


                response_data = {
                    "text": response_text,
                    "gestures": gesture,
                    "port": "50200",
                    "media": None  # Set media to None by default
                }
                print ("kak to zgleda", response_data)
                #dialog_id = get_dialog_id_by_type(category)  # Replace with the actual dialog ID
                dialog_id = get_dialog_id_by_type_and_sender(category, sender_name, email, service_id)

                print("to je ta", dialog_id)
                # Fetch the Dialog object with the matching dialog_id and sender_id
                dialog = Dialog.query.options(joinedload(Dialog.media)).filter_by(id=dialog_id, robot_id=sender.id).first()

                # Fetch associated media for the dialog
                if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    #print("Media file:", media.file)
                    
                    encoded_file = base64.b64encode(media.file).decode('utf-8')  # Convert bytes to base64-encoded string
                    response_data["media"] = {
                        "type": media.type,
                        "file": encoded_file
                    }
                else:
                    print("No media found for the dialog.")
                # TODO skumuniciraj z valentinom
                '''if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    

                    
                    response_data["media"] = {
                        "type": media.type,
                        "media_id": media.id,         # Provide media_id
                        "file_name": media.name,      # Provide file_name
                    }
                else:
                    print("No media found for the dialog.")'''


                # If the service_id is 1 or 2, add language and gender to the response
                if service_id in [1, 2]:
                    response_data["language_code"] = language_id
                    response_data["gender"] = gender

                return jsonify(response_data)
        else:
            '''if is_subscribed_to_fizika == "None":
                if result.get("module"):
                    module_name = result["module"]
                    
                    # Check if the module exists in the Modules table
                    module = Modules.query.filter_by(name=module_name).first()
                    
                    if not module:
                        return jsonify({"error": f"Module '{module_name}' does not exist in the database."}), 400
                    
                    # If the module exists, check the Stripe subscription
                    is_subscribed_to_fizika = check_stripe_subscription(email, module_name)
                    
                    if is_subscribed_to_fizika:
                        print(f"The user is subscribed to {module_name}")
                    else:
                        print(f"The user is NOT subscribed to {module_name}")
                else:
                    print("Module information is missing. Can't verify subscription.")
                print("sadasd alalal", type(is_subscribed_to_fizika))
                if is_subscribed_to_fizika:
                    s.get('http://153.5.66.230:5000/set_session', headers=headers)

                    # Save the session cookies after setting
                    with open(filename, 'wb') as f:
                        pickle.dump(s.cookies, f)
                else:
                    return jsonify({"error": "you need to subscribe"}), 400'''
            

            print("kateri modul imam",  result.get("module"))
            if  result.get("module"):
                module_type = get_module_type_by_name( result.get("module"))
            else:
                return jsonify({"error":"module is not provided"}), 400
            print("kateri modul imam tip", module_type)


            if module_type == "navadni":
                # Fetch the sender object by name, service_id and user id
                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user."}), 404

                # Extract the language and gender if the service_id is 1 or 2
                language_id = sender.language_id if service_id in [1, 2] else None
                gender = sender.gender if service_id in [1, 2] else None

                # Set default conversation state
                if sender not in conversation_states:
                    conversation_states[sender.name] = 'initial'


                recognition_manager = RecognitionManager(sender.name, service_id, email)
                if recognition_manager.check_exit_phrases(text):
                    print("tukaj sem zdaj reset")
                    conversation_states[sender.name] = 'initial'

                category = recognition_manager.get_best_match_category(text)
                print("kaj dobim tu", category)
                dialog_data = get_phrases_response_gesture_by_dialog_type(category, sender_name, email, service_id, result.get("module"))

                if "error" in dialog_data:
                    return jsonify({"error": dialog_data["error"]})

                response_text = ["Nisem razumel."]
                gesture = [""]
                threads = []

                t = threading.Thread(
                    target=process_dialogue_rule,
                    args=(current_app._get_current_object(),category, dialog_data, conversation_states, sender_name, text, response_text, gesture, service_id, email)
                )

                threads.append(t)
                t.start()

                # Wait for all threads to finish
                for t in threads:
                    t.join()
                print("response text", type(response_text))

                if isinstance(response_text, list) and len(response_text) == 1:  # ensure it's a list and has one item
                    sentences = re.split('(?<=[.!?]) +', response_text[0])
                    response_text = [sentence.strip() for sentence in sentences if sentence]


                response_data = {
                    "text": response_text,
                    "gestures": gesture,
                    "port": "50200",
                    "media": None  # Set media to None by default
                }
                print ("kak to zgleda", response_data)
                #dialog_id = get_dialog_id_by_type(category)  # Replace with the actual dialog ID
                dialog_id = get_dialog_id_by_type_and_sender(category, sender_name, email, service_id)

                print("to je ta", dialog_id)
                # Fetch the Dialog object with the matching dialog_id and sender_id
                dialog = Dialog.query.options(joinedload(Dialog.media)).filter_by(id=dialog_id, robot_id=sender.id).first()

                # Fetch associated media for the dialog
                if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    #print("Media file:", media.file)
                    
                    encoded_file = base64.b64encode(media.file).decode('utf-8')  # Convert bytes to base64-encoded string
                    response_data["media"] = {
                        "type": media.type,
                        "file": encoded_file
                    }
                else:
                    print("No media found for the dialog.")
                # TODO skumuniciraj z valentinom
                '''if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    

                    
                    response_data["media"] = {
                        "type": media.type,
                        "media_id": media.id,         # Provide media_id
                        "file_name": media.name,      # Provide file_name
                    }
                else:
                    print("No media found for the dialog.")'''


                # If the service_id is 1 or 2, add language and gender to the response
                if service_id in [1, 2]:
                    response_data["language_code"] = language_id
                    response_data["gender"] = gender

                return jsonify(response_data)
            
            if module_type== "demo":
                # Fetch the sender object by name, service_id and user id
                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user."}), 404

                # Extract the language and gender if the service_id is 1 or 2
                language_id = sender.language_id if service_id in [1, 2] else None
                gender = sender.gender if service_id in [1, 2] else None

                # Set default conversation state
                if sender not in conversation_states:
                    conversation_states[sender.name] = 'initial'


                recognition_manager = RecognitionManager(sender.name, service_id, email)
                if recognition_manager.check_exit_phrases(text):
                    print("tukaj sem zdaj reset")
                    conversation_states[sender.name] = 'initial'

                category = recognition_manager.get_best_match_category(text)
                print("kaj dobim tu", category)
                dialog_data = get_phrases_response_gesture_by_dialog_type(category, sender_name, email, service_id, result.get("module"))

                if "error" in dialog_data:
                    return jsonify({"error": dialog_data["error"]})

                response_text = ["Nisem razumel."]
                gesture = [""]
                threads = []

                t = threading.Thread(
                    target=process_dialogue_rule,
                    args=(current_app._get_current_object(),category, dialog_data, conversation_states, sender_name, text, response_text, gesture, service_id, email)
                )

                threads.append(t)
                t.start()

                # Wait for all threads to finish
                for t in threads:
                    t.join()
                print("response text", type(response_text))

                if isinstance(response_text, list) and len(response_text) == 1:  # ensure it's a list and has one item
                    sentences = re.split('(?<=[.!?]) +', response_text[0])
                    response_text = [sentence.strip() for sentence in sentences if sentence]


                response_data = {
                    "text": response_text,
                    "gestures": gesture,
                    "port": "50200",
                    "media": None  # Set media to None by default
                }
                print ("kak to zgleda", response_data)
                #dialog_id = get_dialog_id_by_type(category)  # Replace with the actual dialog ID
                dialog_id = get_dialog_id_by_type_and_sender(category, sender_name, email, service_id)

                print("to je ta", dialog_id)
                # Fetch the Dialog object with the matching dialog_id and sender_id
                dialog = Dialog.query.options(joinedload(Dialog.media)).filter_by(id=dialog_id, robot_id=sender.id).first()

                # Fetch associated media for the dialog
                if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    #print("Media file:", media.file)
                    
                    encoded_file = base64.b64encode(media.file).decode('utf-8')  # Convert bytes to base64-encoded string
                    response_data["media"] = {
                        "type": media.type,
                        "file": encoded_file
                    }
                else:
                    print("No media found for the dialog.")
                # TODO skumuniciraj z valentinom
                '''if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    

                    
                    response_data["media"] = {
                        "type": media.type,
                        "media_id": media.id,         # Provide media_id
                        "file_name": media.name,      # Provide file_name
                    }
                else:
                    print("No media found for the dialog.")'''


                # If the service_id is 1 or 2, add language and gender to the response
                if service_id in [1, 2]:
                    response_data["language_code"] = language_id
                    response_data["gender"] = gender

                return jsonify(response_data)

            if module_type == "custom":

                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user in custom module."}), 404

                #recognition_manager = RecognitionManager(sender.name, service_id, email)
                recognition_manager_rasa = RecognitionManagerRasa(sender.name, service_id, email)
               
                print("daasds --- tu sem1")
                if recognition_manager_rasa.check_custom_phrases(text):
                    print("imam pravi text", sender_name, "----", text)
                    data_rasa = sendToRasa(sender_name,text, service_id)
                    print("je to to", data_rasa)
                    return data_rasa
                else:
                    return jsonify({"error":"dialog for custom module not in databases"})

            if module_type == "pro chatgpt":

                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user in chatgpt pro module."}), 404


                if sender:
                    print("kaj je text--", text)    
                  
                    # Define the URL of your Flask app
                    url = 'https://spletnastranchat.aibc-robotics.com/search'
                    user_details = getMailPasswordFromUser(sender.name,service_id)
                    if not user_details:
                        print(f"No user details found for sender: {sender.name}")
                        return None  # or handle appropriately
                    jwt = LoginUserChatbot(user_details["name"], user_details["password"])["access_token"]
                    # Define the headers for the request
                    headers = {
                        'Authorization': f'Bearer {jwt}'  # Make sure jwt_token is available in this scope
                    }

                    # Define the query parameter for the request
                    params = {
                        'query': text
                    }

                    response = requests.get(url, headers=headers, params=params)

                    if response.status_code == 200:
                        # Parse the JSON response
                        response_json = response.json()
                        search_result = response_json.get('result', '')  # Extract the result from the response
                    else:
                        print(f'Error: Unable to retrieve data. HTTP Status Code: {response.status_code}')
                        search_result = 'Error: Unable to retrieve data.'  # Default error message


                    # Construct the desired response shape
                    result = {
                        "gestures": ["tega ni"],  # example static gesture value
                        "media": None,       # example static media value
                        "port": "50200",     # example static port value
                        "text": [search_result]  # using the response from the external API
                    }

                    return jsonify(result)
                else:
                    return jsonify({"error": "dialog for custom module not in databases"})

            
            if module_type == "pro učbeniki":

                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user in chatgpt pro module."}), 404

                if sender:
                    print("kaj je text--", text, service_id)    
                    
                    user_details = getMailPasswordFromUser(sender.name,service_id)
                    print("kaj dobim tu,", sender, user_details)
                    if not user_details:
                        print(f"No user details found for sender: {sender.name}")
                        return None  # or handle appropriately
                    jwt = LoginUserChatbot(user_details["name"], user_details["password"])["access_token"]


                    # Fetch subject and education (subject_id) from the database for sender
                    subject = sender.subject
                    education = sender.subject_id

                    if not subject or not education:
                        return jsonify({"error": "Subject or education information missing for the sender."}), 404

                    result_query_api = query_api(text, subject, education,jwt)


                    # Construct the desired response shape
                    result = {
                        "gestures": ["tega ni"],  # example static gesture value
                        "media": None,       # example static media value
                        "port": "50200",     # example static port value
                        "text": [result_query_api]  # using the response from the new query_api function
                    }

                    return jsonify(result)
                else:
                    return jsonify({"error": "dialog for custom module not in databases"})

            if module_type == "chatgpt":
                # Fetch the sender object by name, service_id and user id
                sender = User_robot.query.filter_by(name=sender_name, service_id=service_id, sender_user_id=user.id).first()
                if sender is None:
                    return jsonify({"error": "Sender not found for the given service or user."}), 404

                # Extract the language and gender if the service_id is 1 or 2
                language_id = sender.language_id if service_id in [1, 2] else None
                gender = sender.gender if service_id in [1, 2] else None

                # Set default conversation state
                if sender not in conversation_states:
                    conversation_states[sender.name] = 'initial'


                recognition_manager = RecognitionManager(sender.name, service_id, email)
                if recognition_manager.check_exit_phrases(text):
                    print("tukaj sem zdaj reset")
                    conversation_states[sender.name] = 'initial'

                category = recognition_manager.get_best_match_category(text)
                print("kaj dobim tu", category)
                dialog_data = get_phrases_response_gesture_by_dialog_type(category, sender_name, email, service_id, result.get("module"))

                if "error" in dialog_data:
                    if dialog_data["error"] == 'No dialog found for the given dialog type. Please add dialogs or intents.':
                        # Call OpenAI API to get a response
                        refined_prompt = "Ti si pogovorni asistent, ki odgovarjaš na vprašanja in podajaš čimbolj pravilne odgovore. " + text
                        chat_gpt_response = openai.Completion.create(
                            engine="text-davinci-003",
                            prompt=refined_prompt,
                            max_tokens=150,
                            temperature=0.2
                        )
                        response = chat_gpt_response.choices[0].text.strip()

                        # Convert the response into a list of sentences/fragments
                        sentences = response.split('\n')

                        # Remove any standalone '!' from the list of sentences
                        sentences = [s for s in sentences if s != '!']

                        # Join the sentences into a single string
                        joined_response = ' '.join(sentences).strip()

                        # Remove any unwanted patterns (like HTML tags, if any are present)
                        cleaned_response = re.sub('<.*?>', '', joined_response)

                        # Append the note to indicate the response was generated by ChatGPT
                        final_response = cleaned_response + " (odgovor podal ChatGPT)"
                        response_text = [final_response]
                    else:
                        return jsonify({"error": dialog_data["error"]})

                response_text = ["Nisem razumel."]
                gesture = [""]
                threads = []

                t = threading.Thread(
                    target=process_dialogue_rule,
                    args=(current_app._get_current_object(),category, dialog_data, conversation_states, sender_name, text, response_text, gesture, service_id, email)
                )

                threads.append(t)
                t.start()

                # Wait for all threads to finish
                for t in threads:
                    t.join()
                print("response text-----------", response_text)

                if not response_text or response_text == ["Nisem razumel."]:  # check if the response_text is not found or default
                    # Call OpenAI API to get a response
                    print("sem sploh prideš ti?----")
                    refined_prompt = "Ti si pogovorni asistent, ki odgovarjaš na vprašanja in podajaš čimbolj pravilne odgovore. " + text
                    chat_gpt_response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=refined_prompt,
                        max_tokens=150,
                        temperature=0.2
                    )

                    response = chat_gpt_response.choices[0].text.strip()

                    # Convert the response into a list of sentences/fragments
                    sentences = response.split('\n')

                    # Remove any standalone '!' from the list of sentences
                    sentences = [s for s in sentences if s != '!']

                    # Join the sentences into a single string
                    joined_response = ' '.join(sentences).strip()

                    # Remove any unwanted patterns (like HTML tags, if any are present)
                    cleaned_response = re.sub('<.*?>', '', joined_response)

                    # Append the note to indicate the response was generated by ChatGPT
                    final_response = cleaned_response + " (odgovor podal ChatGPT)"
                    response_text = [final_response]
                
                if isinstance(response_text, list) and len(response_text) == 1:  # ensure it's a list and has one item
                    sentences = re.split('(?<=[.!?]) +', response_text[0])
                    response_text = [sentence.strip() for sentence in sentences if sentence]


                response_data = {
                    "text": response_text,
                    "gestures": gesture,
                    "port": "50200",
                    "media": None  # Set media to None by default
                }
                print ("kak to zgleda", response_data)
                #dialog_id = get_dialog_id_by_type(category)  # Replace with the actual dialog ID
                dialog_id = get_dialog_id_by_type_and_sender(category, sender_name, email, service_id)

                print("to je ta", dialog_id)
                # Fetch the Dialog object with the matching dialog_id and sender_id
                dialog = Dialog.query.options(joinedload(Dialog.media)).filter_by(id=dialog_id, robot_id=sender.id).first()

                # Fetch associated media for the dialog
                if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    #print("Media file:", media.file)
                    
                    encoded_file = base64.b64encode(media.file).decode('utf-8')  # Convert bytes to base64-encoded string
                    response_data["media"] = {
                        "type": media.type,
                        "file": encoded_file
                    }
                else:
                    print("No media found for the dialog.")
                # TODO skumuniciraj z valentinom
                '''if dialog and dialog.media:
                    media = dialog.media
                    print("Media type:", media.type)
                    

                    
                    response_data["media"] = {
                        "type": media.type,
                        "media_id": media.id,         # Provide media_id
                        "file_name": media.name,      # Provide file_name
                    }
                else:
                    print("No media found for the dialog.")'''


                # If the service_id is 1 or 2, add language and gender to the response
                if service_id in [1, 2]:
                    response_data["language_code"] = language_id
                    response_data["gender"] = gender

                return jsonify(response_data)

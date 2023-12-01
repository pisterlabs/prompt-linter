import os
import requests
import json
import time
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import requests
import json
chat = ChatOpenAI(temperature=0,model_name="gpt-4")

BASE_URL = "https://apitorgi.myseldon.com"
LOGIN_ENDPOINT = "/User/Login"
CREATE_PURCHASE_REQUEST_ENDPOINT = "/Purchases/New"  
FETCH_PURCHASES_RESULTS_ENDPOINT = "/Purchases/Result"

def login_to_service():
    login_url = f"{BASE_URL}{LOGIN_ENDPOINT}"
    
    payload = {
        "name": os.getenv("MYSELDON_LOGIN"),
        "password": os.getenv("MYSELDON_PASS")
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.post(login_url, json=payload, headers=headers)
    if response.status_code == 200 and 'result' in response.json() and 'token' in response.json()['result']:
        return response.json()['result']['token']
    else:
        raise ValueError("Failed to login!")

def request_purchase_data(token):
    purchase_request_url = f"{BASE_URL}{CREATE_PURCHASE_REQUEST_ENDPOINT}?token={token}"
    
    date_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"
    date_to = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"

    payload = {
        "filterId": 4349670,
        "dateFrom": date_ago,
        "dateTo": date_to,
        "subtype": 0
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.post(purchase_request_url, json=payload, headers=headers)
    response_data = response.json()
    if response.status_code == 200 and 'result' in response_data and 'taskId' in response_data['result']:
        return response_data['result']['taskId']
    else:
        raise ValueError("Failed to request purchase data!")

def check_task_status(token, task_id):
    check_status_url = f"{BASE_URL}/Purchases/Status?token={token}"
    payload = {
        "taskId": task_id
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.post(check_status_url, json=payload, headers=headers)
    if response.status_code == 200:
        status_code = response.json()['result']['searchStatus']['code']
        quantity = response.json()['result']['quantity']
        if status_code == 3:
            print(f"Task ID: {task_id} is ready with quantity: {quantity}")
        return status_code == 3, quantity
    else:
        print(f"Error checking task status: {response.json()}")
        return False, 0

def fetch_purchases_results(token, task_id):
    # Poll the server for task completion
    is_ready, quantity = check_task_status(token, task_id)
    while not is_ready:
        time.sleep(10)  # wait for a short period before checking again
        is_ready, quantity = check_task_status(token, task_id)

    if quantity == 0:
        print("Quantity is 0, finishing function.")
        return

    result_url = f"{BASE_URL}{FETCH_PURCHASES_RESULTS_ENDPOINT}?token={token}"
    payload = {
        "taskId": task_id,
        "pageIndex": 1
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    #load prompt.txt
    with open("prompt.txt", "r") as f:
        prompt = f.read()
        f.close()
    try:
        response = requests.post(result_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise ValueError("Failed to fetch purchases results!")

        # Debug print
        #print(f"Response JSON: {response.json()}")

        purchase_data = response.json()
        if 'result' not in purchase_data or 'purchases' not in purchase_data['result']:
            raise ValueError("Unexpected response format. 'result' or 'purchases' key is missing.")
        
        purchases = purchase_data['result']['purchases']

        if not os.path.exists('purchases'):
            os.makedirs('purchases')

        for purchase in purchases:
            seldon_id = purchase['SeldonId']
            purchasetext = json.dumps(purchase, ensure_ascii=False)
            
            print(seldon_id)

            with open(f'purchases/{seldon_id}.json', 'w') as outfile:
                json.dump(purchase, outfile)
            
            print(len(purchasetext))
            if (len(purchasetext) > 30000): 
                print(" Too long ")
                #remove lotsList field from purchase
                purchase.pop('lotsList', None)
                purchasetext = json.dumps(purchase, ensure_ascii=False)
                print(len(purchasetext))

            messages = [
                SystemMessage(
                    content=prompt
                ),
                HumanMessage(
                    content=purchasetext
                )
            ]

            gpttitle = chat(messages)
            #wait 60 sec
            time.sleep(20)

            print(gpttitle.content)
            

            if gpttitle.content is not None and len(gpttitle.content) > 20: # and "Не подходит" not in gpttitle.content
                
                gpttitle.content = gpttitle.content.replace("&", "%26")
                webhook_url = f"https://noxon.wpmix.net/counter.php?totenders=1&msg={gpttitle.content}"              
                response = requests.post(webhook_url)
                if response.status_code != 200:
                    print(f"Failed to send webhook for tender at index {idx}. Status code: {response.status_code}")


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    token = login_to_service()
    task_id = request_purchase_data(token)
    fetch_purchases_results(token, task_id)

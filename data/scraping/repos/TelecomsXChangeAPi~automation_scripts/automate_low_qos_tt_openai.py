"""
Script Name: Telecomsxchange Trouble Ticketing
Version: 1.0
Last Modified: 2023-06-12
Author: Ameed Jamous

Description:
This script fetches the call history data from Telecomsxchange and NeuTrafix buyer's API and identifies failed calls 
based on certain disconnect reasons. For each failed call identified, it leverages OpenAI's GPT-3 model 
to generate a professional message to notify the respective vendor. The script ensures that a message 
for a particular call is only sent once by keeping a record of sent messages in a text file. 

This script also includes logging functionality which records each step of the script execution, both 
in a log file ("debug.log") and in the console. 

Credentials, API endpoints, and other configurations can be adjusted in the respective sections of the 
script.
"""



import requests
from requests.auth import HTTPDigestAuth
import json
from datetime import datetime, timedelta
import os
import random
import string
import openai
from openai.api_resources.completion import Completion
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler("debug.log"),
                              logging.StreamHandler()])

logging.info('Script started.')

# Function to generate random ticket ID
def get_random_ticket_id(length):
    letters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

# Functions to handle sent messages
def check_if_message_sent(call_id, destination_number, vendor_id):
    message_id = f'{call_id}-{destination_number}-{vendor_id}'
    if not os.path.exists('sent_messages.txt'):
        return False
    with open('sent_messages.txt', 'r') as f:
        for line in f.readlines():
            if line.strip() == message_id:
                return True
    return False

def record_message_sent(call_id, destination_number, vendor_id):
    message_id = f'{call_id}-{destination_number}-{vendor_id}'
    with open('sent_messages.txt', 'a') as f:
        f.write(message_id + '\n')

# Set your OpenAI key
openai.api_key = 'openapi-key'

# Function to generate message with OpenAI
def generate_message(cdr):
    prompt = f"We need to inform a vendor/seller about a call failure in professional written way. The call to destination number {cdr['CLD']} on route {cdr['connection_name']} failed. The disconnect reason was: {cdr['disconnect_reason']}. The timestamp of occurrence was: {cdr['connect_time']}, Ask them to confirm when the issue has been fixed and do NOT end the message with Best Regards, name etc.."
    message_completion = Completion.create(model="text-davinci-002", prompt=prompt, max_tokens=500)
    return message_completion.choices[0].text.strip()

# Telecomsxchange or NeuTrafix Buyer Credentials
username = '{Username}'
password = '{API-Key}'

# API Endpoints (TCXC)
call_history_url = 'https://apiv2.telecomsxchange.com/buyers/callhistory/'
message_send_url = 'https://apiv2.telecomsxchange.com/buyers/message/send'

# API Endpoints (NeuTrafix)
# call_history_url = 'https://apiv2.neutrafix.telin.net/buyers/callhistory/'
# message_send_url = 'https://apiv2.neutrafix.telin.net/buyers/message/send'


# Headers for the requests
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}

# Time settings for the call history API request
end_time = datetime.utcnow()
start_time = end_time - timedelta(minutes=5440)

# Data for the call history API request
call_history_data = {
    'show': 'bad',
    'date_from': start_time.strftime('%Y-%m-%d %H:%M:%S'),
    'date_to': end_time.strftime('%Y-%m-%d %H:%M:%S'),
}

# Disconnect reasons to filter out
disconnect_reasons = ["Service or option not available", "Call rejected", "timeout", "Internetworking, unspecified", "Bearer capability not authorized", "unspecified"]

# Send GET request for call history
call_history_response = requests.post(call_history_url, headers=headers, data=call_history_data,
                                      auth=HTTPDigestAuth(username, password))

# Check if the call history request was successful
if call_history_response.status_code == 200:
    try:
        call_history_info = call_history_response.json()
        if call_history_info.get('status') == 'success':
            logging.info('Call history request was successful.')
            cdrs = call_history_info.get('cdrs')
            for cdr in cdrs:
                disconnect_reason = cdr.get('disconnect_reason')
                if disconnect_reason in disconnect_reasons and not check_if_message_sent(cdr['call_id'], cdr['CLD'], cdr['i_vendor']):
                    logging.info(f'Call failure detected for call id {cdr["call_id"]}, destination {cdr["CLD"]}, vendor {cdr["i_vendor"]}.')
                    generated_message = generate_message(cdr)
                    message_data = {
                        'id': cdr['i_vendor'],
                        'subject': f"Trouble Ticket #{get_random_ticket_id(10)}: Noted Call Failure to {cdr['CLD']}",
                        'message': generated_message
                    }
                    message_response = requests.post(message_send_url, headers=headers, data=message_data,
                                                      auth=HTTPDigestAuth(username, password))
                    record_message_sent(cdr['call_id'], cdr['CLD'], cdr['i_vendor'])
        else:
            logging.error("Call history request returned an error. Details: ", call_history_info)
    except ValueError:
        logging.error("Error decoding the call history response as JSON")
else:
    logging.error("Call history request failed")

logging.info('Script finished.')
